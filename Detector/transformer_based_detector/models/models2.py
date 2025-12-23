import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer

# ---------- Small building blocks ----------

class FiLMSpeakerConditioner(nn.Module):
    """Applies FiLM conditioning from a speaker embedding: (1+γ) * h + β."""
    def __init__(self, d_model, speaker_dim, normalize_emb=True, dropout_p=0.0):
        super().__init__()
        self.affine = nn.Linear(speaker_dim, 2 * d_model)
        self.normalize_emb = normalize_emb
        self.dropout_p = dropout_p

    def forward(self, context, speaker_emb):  # context: [B, T, D], emb: [B, E]
        if self.normalize_emb:
            speaker_emb = F.normalize(speaker_emb, dim=-1)
        if self.training and self.dropout_p > 0:
            keep = (torch.rand(speaker_emb.size(0), 1, device=speaker_emb.device) > self.dropout_p).float()
            speaker_emb = speaker_emb * keep
        gamma, beta = self.affine(speaker_emb).chunk(2, dim=-1)  # [B, D] each
        return (1 + gamma).unsqueeze(1) * context + beta.unsqueeze(1)  # [B, T, D]


class SpeakerBiasAdapter(nn.Module):
    """Simple additive bias from speaker emb: x + W_e(emb)."""
    def __init__(self, d_model, speaker_dim, normalize_emb=True, dropout_p=0.0):
        super().__init__()
        self.proj = nn.Linear(speaker_dim, d_model)
        self.normalize_emb = normalize_emb
        self.dropout_p = dropout_p

    def forward(self, x, speaker_emb):  # x: [B,T,D], emb: [B,E]
        if self.normalize_emb:
            speaker_emb = F.normalize(speaker_emb, dim=-1)
        if self.training and self.dropout_p > 0:
            keep = (torch.rand(speaker_emb.size(0), 1, device=speaker_emb.device) > self.dropout_p).float()
            speaker_emb = speaker_emb * keep
        b = self.proj(speaker_emb).unsqueeze(1)  # [B,1,D]
        return x + b


# ---------- Encoders ----------

class ConformerEncoder(nn.Module):
    """
    Projects log-mel then runs a Conformer encoder.
    Optional FiLM conditioning BEFORE the Conformer so the encoder is speaker-aware.
    """
    def __init__(self, input_dim=64, encoder_dim=256,
                 speaker_dim=None, film_in_encoder=False, emb_dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, encoder_dim)
        self.conformer = Conformer(
            input_dim=encoder_dim,
            num_heads=4,
            ffn_dim=encoder_dim * 2,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )
        self.film = None
        if film_in_encoder:
            if speaker_dim is None:
                raise ValueError("speaker_dim must be set when film_in_encoder=True")
            self.film = FiLMSpeakerConditioner(
                d_model=encoder_dim, speaker_dim=speaker_dim, normalize_emb=True, dropout_p=emb_dropout
            )

    def forward(self, x, lengths=None, speaker_emb=None):  # x: [B,T,F]
        x = self.proj(x)  # [B,T,D]
        if self.film is not None and speaker_emb is not None:
            x = self.film(x, speaker_emb)  # early conditioning
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        out, _ = self.conformer(x, lengths)  # (some versions return [B,D,T])
        if out.dim() == 3 and out.shape[1] != x.shape[1] and out.shape[2] == x.shape[1]:
            out = out.transpose(1, 2)
        return out  # [B,T,D]


class CausalTransformer(nn.Module):
    """Causal TransformerEncoder with proper key-padding masking."""
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1, causal=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.causal = causal

    def forward(self, x, lengths=None):  # [B,T,D]
        B, T, _ = x.shape
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = None
        if lengths is not None:
            idxs = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            key_padding_mask = idxs >= lengths.unsqueeze(1)
        return self.transformer(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)


# ---------- Heads ----------

class SpeakerConditionedClassifier(nn.Module):
    """FiLM-conditioned per-frame classifier. Returns LOGITS (no sigmoid)."""
    def __init__(self, d_model, speaker_dim, normalize_emb=True, emb_dropout=0.0):
        super().__init__()
        self.film = FiLMSpeakerConditioner(
            d_model=d_model, speaker_dim=speaker_dim, normalize_emb=normalize_emb, dropout_p=emb_dropout
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, context, speaker_emb):  # [B,T,D], [B,E]
        x = self.film(context, speaker_emb)
        return self.head(x).squeeze(-1)  # [B,T]


# ---------- Full model ----------

class SpeakerSuppressionDetector(nn.Module):
    """
    End-to-end framewise suppression detector conditioned on a target speaker.
    Early + mid + late conditioning:
      - FiLM in Conformer input (optional)
      - Speaker-bias into context net input (cheap & robust)
      - FiLM in predictor (as before)
    """
    def __init__(self, input_dim=64, encoder_dim=144, speaker_dim=192,
                 nhead=4, num_layers=2, emb_dropout=0.1,
                 film_in_encoder=True,     # NEW: enable early conditioning
                 add_context_bias=True,    # NEW: mid-layer conditioning
                 context_bias_dropout=0.1):
        super().__init__()
        # Early: encoder FiLM
        self.encoder = ConformerEncoder(
            input_dim=input_dim, encoder_dim=encoder_dim,
            speaker_dim=speaker_dim, film_in_encoder=film_in_encoder, emb_dropout=emb_dropout
        )
        # Mid: light bias adapter before causal transformer
        self.context_adapter = SpeakerBiasAdapter(
            d_model=encoder_dim, speaker_dim=speaker_dim,
            normalize_emb=True, dropout_p=context_bias_dropout
        ) if add_context_bias else None

        self.context_net = CausalTransformer(d_model=encoder_dim, nhead=nhead, num_layers=num_layers)

        # Late: FiLM head (unchanged)
        self.predictor = SpeakerConditionedClassifier(
            d_model=encoder_dim, speaker_dim=speaker_dim, normalize_emb=True, emb_dropout=emb_dropout
        )

    @staticmethod
    def infer_lengths_from_padded_logmel(logmel):  # logmel: [B, T, F]
        return (logmel.abs().sum(dim=-1) > 0).sum(dim=1).to(dtype=torch.long)

    def forward(self, logmel, speaker_emb, lengths=None):
        if lengths is None:
            lengths = self.infer_lengths_from_padded_logmel(logmel)
        # Early conditioning inside encoder
        z = self.encoder(logmel, lengths=lengths, speaker_emb=speaker_emb)  # [B,T,D]

        # Mid conditioning (cheap bias from speaker)
        if self.context_adapter is not None:
            z = self.context_adapter(z, speaker_emb)  # [B,T,D]

        c = self.context_net(z, lengths=lengths)      # [B,T,D]
        logits = self.predictor(c, speaker_emb)       # [B,T]
        return logits
