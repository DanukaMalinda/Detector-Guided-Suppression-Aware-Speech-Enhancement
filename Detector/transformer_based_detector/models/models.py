import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer


class ConformerEncoder(nn.Module):
    """Projects log-mel frames then runs a Conformer encoder."""
    def __init__(self, input_dim=64, encoder_dim=256):
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

    def forward(self, x, lengths=None):  # x: [B, T, F]
        x = self.proj(x)  # [B, T, D]
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)

        out, _ = self.conformer(x, lengths)  # some versions may return [B, D, T]
        # ---- Ensure [B, T, D] ----
        if out.dim() == 3 and out.shape[1] != x.shape[1] and out.shape[2] == x.shape[1]:
            # got [B, D, T] → transpose to      [B, T, D]
            out = out.transpose(1, 2)
        return out  # [B, T, D]


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

    def forward(self, x, lengths=None):  # x: [B, T, D]
        B, T, _ = x.shape

        # Causal (look-ahead) mask: True means "mask out"
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )

        # Key padding mask: True at PAD positions
        key_padding_mask = None
        if lengths is not None:
            idxs = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            key_padding_mask = idxs >= lengths.unsqueeze(1)  # [B, T], True=pad

        return self.transformer(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)


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
            # Embedding dropout: forces the net to learn to use (but not rely solely on) the emb
            keep = (torch.rand(speaker_emb.size(0), 1, device=speaker_emb.device) > self.dropout_p).float()
            speaker_emb = speaker_emb * keep

        gamma_beta = self.affine(speaker_emb)  # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, D] each
        return (1 + gamma).unsqueeze(1) * context + beta.unsqueeze(1)  # [B, T, D]


class SpeakerConditionedClassifier(nn.Module):
    """FiLM-conditioned per-frame classifier. Returns LOGITS (no sigmoid)."""
    def __init__(self, d_model, speaker_dim, normalize_emb=True, emb_dropout=0.0):
        super().__init__()
        self.film = FiLMSpeakerConditioner(
            d_model=d_model,
            speaker_dim=speaker_dim,
            normalize_emb=normalize_emb,
            dropout_p=emb_dropout,
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, context, speaker_emb):  # [B, T, D], [B, E]
        x = self.film(context, speaker_emb)       # [B, T, D]
        logits = self.head(x).squeeze(-1)         # [B, T]
        return logits


class SpeakerSuppressionDetector(nn.Module):
    """
    End-to-end framewise suppression detector conditioned on a target speaker.
    Returns logits; use BCEWithLogitsLoss in training.
    """
    def __init__(self, input_dim=64, encoder_dim=144, speaker_dim=192,
                 nhead=4, num_layers=2, emb_dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim=input_dim, encoder_dim=encoder_dim)
        self.context_net = CausalTransformer(d_model=encoder_dim, nhead=nhead, num_layers=num_layers)
        self.predictor = SpeakerConditionedClassifier(
            d_model=encoder_dim, speaker_dim=speaker_dim, normalize_emb=True, emb_dropout=emb_dropout
        )

    @staticmethod
    def infer_lengths_from_padded_logmel(logmel):  # logmel: [B, T, F]
        # counts non-zero frames; adjust if you use a real length tensor
        return (logmel.abs().sum(dim=-1) > 0).sum(dim=1).to(dtype=torch.long)

    def forward(self, logmel, speaker_emb, lengths=None):
        """
        logmel: [B, T, F]
        speaker_emb: [B, E] (ECAPA = 192 by default)
        lengths: Optional [B] true lengths in frames (before padding)
        """
        if lengths is None:
            lengths = self.infer_lengths_from_padded_logmel(logmel)

        z = self.encoder(logmel, lengths=lengths)       # [B, T, D]
        c = self.context_net(z, lengths=lengths)        # [B, T, D] with padding masked
        logits = self.predictor(c, speaker_emb)         # [B, T] (LOGITS)
        return logits
