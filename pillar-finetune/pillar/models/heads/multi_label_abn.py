import torch
import torch.nn as nn

from pillar.models.abstract_model import AbstractModel


class MultiLabelClassificationHead(AbstractModel):
    def __init__(
        self,
        args,
        input_dim: int = None,
        num_classes: int = 18,
        dropout: float = 0.0,
        hidden_dim: int = 0,
        **extras,
    ):
        super(MultiLabelClassificationHead, self).__init__(args)

        assert input_dim is not None, "input_dim must be provided"

        self.num_classes = num_classes

        # hidden_dim=0 (default): plain Linear probe head, unchanged from before.
        # hidden_dim>0: single-hidden-layer MLP (matches the architecture that won in
        # scripts/train_abn_probe.py's frozen-feature probing, e.g. hidden_dim=1024).
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(input_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, input_dim)  — backbone feature

        Returns:
            logits : (B, num_classes)  — raw logits (sigmoid는 loss에서 처리)
        """
        return self.head(x)  # (B, 18)