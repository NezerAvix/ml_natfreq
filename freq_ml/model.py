"""
PointNet-подобная регрессия собственных частот (позиция + нормали, attention-pool)
"""

from __future__ import annotations

import torch
from torch import nn

from freq_ml.constants import N_FREQ
from freq_ml.geometry import N_SCALARS, POINT_DIM


class FrequencyPredictor(nn.Module):
    def __init__(
        self,
        n_points: int = 2048,
        n_freq: int = N_FREQ,
        n_scalars: int = N_SCALARS,
        point_dim: int = POINT_DIM,
        width: int = 512,
    ):
        super().__init__()
        self.width = width
        self.point_dim = point_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(width, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.scalar_bn = nn.BatchNorm1d(n_scalars)
        self.head = nn.Sequential(
            nn.Linear(width + n_scalars, width),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_freq),
        )

    def forward(self, points: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b, n, c = points.shape
        x = points.reshape(b * n, c)
        x = self.point_mlp(x).reshape(b, n, -1)
        scores = self.attn_score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        x_max = torch.max(x, dim=1).values
        fused = 0.5 * pooled + 0.5 * x_max
        s = self.scalar_bn(scalars)
        return self.head(torch.cat([fused, s], dim=1))
