"""Minimal, dependency-trimmed vendoring of pillar-finetune's retrieval head.

Only Simple_TranSQ (+ the DETR decoder and 3D sincos pos-embed it uses) is
vendored; importing the full `pillar` package would drag in kornia/flash_attn.
Import rewrites are done by sync_sources.sh — do not edit these files by hand.
"""
from .transq import Simple_TranSQ  # noqa: F401
