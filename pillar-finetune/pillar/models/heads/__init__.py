"""Head models."""

from pillar.models.heads.cumulative_hazard_layer import (
    CumulativeProbabilityLayer,
)
from pillar.models.heads.detr import DETR3D
from pillar.models.heads.multilabel_linear import (
    MultiLabelLinearHead,
)
from pillar.models.heads.multi_label_abn import MultiLabelClassificationHead
from pillar.models.heads.transq import Simple_TranSQ

__all__ = [
    "CumulativeProbabilityLayer",
    "DETR3D",
    "MultiLabelLinearHead",
    "MultiLabelClassificationHead",
    "Simple_TranSQ",
]