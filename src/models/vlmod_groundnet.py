from __future__ import annotations

import re

import torch
from torch import nn


class DescriptionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=512,
            embedding_dim=hidden_dim,
        )

        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def tokenize(self, text: str, max_length: int = 48) -> list[int]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        ids = []

        for token in tokens[:max_length]:
            value = sum(ord(char) for char in token) % 510
            ids.append(value + 1)

        if not ids:
            ids = [0]

        ids.extend([0] * (max_length - len(ids)))
        return ids

    def forward(self, descriptions: list[str]) -> torch.Tensor:
        token_ids = torch.tensor(
            [self.tokenize(text) for text in descriptions],
            dtype=torch.long,
            device=self.embedding.weight.device,
        )

        embedded = self.embedding(token_ids)
        encoded, _ = self.encoder(embedded)

        pooled = encoded.mean(dim=1)
        return self.projection(pooled)


class VLMODGroundNet(nn.Module):
    def __init__(
        self,
        object_dim: int = 20,
        hidden_dim: int = 128,
        num_descriptions: int = 3,
    ):
        super().__init__()

        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.description_encoder = DescriptionEncoder(hidden_dim)

        self.description_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

        self.num_descriptions = num_descriptions

    def forward(
        self,
        object_features: torch.Tensor,
        descriptions: list[str],
    ) -> torch.Tensor:
        object_tokens = self.object_encoder(object_features)
        text_tokens = self.description_encoder(descriptions)
        text_tokens = self.description_projection(text_tokens)

        num_objects = object_tokens.size(0)

        object_expanded = object_tokens.unsqueeze(1).expand(
            num_objects,
            self.num_descriptions,
            -1,
        )

        text_expanded = text_tokens.unsqueeze(0).expand(
            num_objects,
            -1,
            -1,
        )

        combined = torch.cat(
            [object_expanded, text_expanded],
            dim=-1,
        )

        gate = self.fusion_gate(combined)
        gated_text = gate * text_expanded

        fused = torch.cat(
            [object_expanded, gated_text],
            dim=-1,
        )

        return self.prediction_head(fused).squeeze(-1)
