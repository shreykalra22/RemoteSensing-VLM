"""A trainable, modality-separated cross-modal Transformer classifier."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FusionConfig:
    num_labels: int = 19
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    text_vocab_size: int = 4096
    max_text_tokens: int = 32


class ImageTokenEncoder(nn.Module):
    """Small trainable image encoder that emits a fixed token sequence."""

    def __init__(self, channels: int, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, dimension // 2, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(dimension // 2, dimension, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected [batch, channels, height, width], got {tuple(image.shape)}")
        features = self.network(image)
        return features.flatten(2).transpose(1, 2)


class MetadataTextEncoder(nn.Module):
    """Trainable encoder for non-target scene metadata tokens.

    Callers must not tokenize ground-truth land-cover labels when those labels
    are the prediction target.
    """

    def __init__(self, vocab_size: int, dimension: int, max_tokens: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, dimension))

    def forward(self, token_ids: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        if token_ids is None:
            return None, None
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_tokens:
            raise ValueError(f"Expected text tokens [batch, <= {self.max_tokens}], got {tuple(token_ids.shape)}")
        padding = token_ids.eq(0)
        return self.embedding(token_ids) + self.position[:, : token_ids.shape[1]], padding


class CrossModalFusionClassifier(nn.Module):
    """S1/S2/token fusion using trainable Transformer self-attention.

    S1 and S2 produce distinct image tokens. Text is optional; a learned
    modality token represents a missing text input without inventing content.
    """

    def __init__(self, config: FusionConfig = FusionConfig()) -> None:
        super().__init__()
        if config.embedding_dim % config.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.config = config
        self.s1_encoder = ImageTokenEncoder(2, config.embedding_dim)
        self.s2_encoder = ImageTokenEncoder(10, config.embedding_dim)
        self.text_encoder = MetadataTextEncoder(config.text_vocab_size, config.embedding_dim, config.max_text_tokens)
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.s1_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.s2_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.no_text_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(config.embedding_dim, config.heads, config.embedding_dim * 4, config.dropout, batch_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.classifier = nn.Linear(config.embedding_dim, config.num_labels)

    def forward(self, s1: Tensor, s2: Tensor, text_tokens: Tensor | None = None) -> Tensor:
        if s1.ndim != 4 or s1.shape[1] != 2:
            raise ValueError("Sentinel-1 input must have shape [batch, 2, height, width]")
        if s2.ndim != 4 or s2.shape[1] != 10:
            raise ValueError("Sentinel-2 input must have shape [batch, 10, height, width]")
        if s1.shape[0] != s2.shape[0]:
            raise ValueError("Sentinel-1 and Sentinel-2 batch sizes must match")
        batch = s1.shape[0]
        text, text_padding = self.text_encoder(text_tokens)
        prefix = torch.cat((self.class_token, self.s1_token, self.s2_token), dim=1).expand(batch, -1, -1)
        image_tokens = torch.cat((self.s1_encoder(s1), self.s2_encoder(s2)), dim=1)
        if text is None:
            text = self.no_text_token.expand(batch, -1, -1)
            text_padding = torch.zeros((batch, 1), dtype=torch.bool, device=s1.device)
        tokens = torch.cat((prefix, image_tokens, text), dim=1)
        padding = torch.cat((torch.zeros((batch, prefix.shape[1] + image_tokens.shape[1]), dtype=torch.bool, device=s1.device), text_padding.to(s1.device)), dim=1)
        return self.classifier(self.fusion(tokens, src_key_padding_mask=padding)[:, 0])
