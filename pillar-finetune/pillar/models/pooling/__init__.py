"""Pooling modules."""

from pillar.models.pooling.multi_attention_pool_layers import MultiAttentionPool
from pillar.models.pooling.volume_attention_pool import VolumeAttentionPool
from pillar.models.pooling.basic_pool_layers import GlobalMaxPool

__all__ = ["MultiAttentionPool", "VolumeAttentionPool", "GlobalMaxPool"]
