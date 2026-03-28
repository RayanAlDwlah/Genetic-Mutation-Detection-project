"""1D CNN model for mutation classification.

NOTE: This module is a structural placeholder reserved for Phase 2 (post-midterm).
The full implementation will include:
  - Convolutional feature extraction over encoded amino acid sequences
  - Multi-head self-attention layer for long-range residue interactions
  - Training loop with class-weighted loss, early stopping, and AUC evaluation

Current class: SimpleMutationCNN — minimal 1D CNN used for interface validation only.
"""

import torch
from torch import nn


class SimpleMutationCNN(nn.Module):
    """Minimal 1D CNN baseline for sequence-based mutation inputs.

    Architecture: two Conv1d layers with ReLU activations, max pooling,
    global average pooling, and a linear classifier head.
    This stub is used to verify the data pipeline and import structure.
    The production model will replace this with a deeper architecture.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)
