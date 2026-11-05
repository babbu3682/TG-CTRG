"""Backbone models."""

from pillar.models.backbones.mmatlas import MultimodalAtlas
# mmatlas_self.py defines its own standalone `MultimodalAtlas` class (loads raw .pt
# checkpoints via a "visual."-prefixed state_dict, not HF AutoModel) — same class name,
# different module, so alias it here to avoid colliding with the HF-backed one above.
from pillar.models.backbones.mmatlas_self import MultimodalAtlas as MultimodalAtlasSelf
from pillar.models.backbones.mmatlas_self_vlcabs import MultimodalAtlasVLCABS
from pillar.models.backbones.mmatlas_self_lapa import MultimodalAtlasSelfLAPA

__all__ = ["MultimodalAtlas", "MultimodalAtlasSelf", "MultimodalAtlasVLCABS", "MultimodalAtlasSelfLAPA"]
