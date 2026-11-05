import torch.nn as nn

from pillar.models.abstract_model import AbstractModel


class MultiLabelLinearHead(AbstractModel):
    def __init__(
        self,
        args,
        input_dim,
        output_dim=18,
        dropout_rate=0.0,
        **extras,
    ):
        super().__init__(args)

        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate > 0
            else nn.Identity()
        )

        self.fc = nn.Linear(
            input_dim,
            output_dim,
        )

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)