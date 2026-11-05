"""Minimal stand-in for pillar.utils.logging — the vendored backbone only
calls logger.info/warning."""
import logging
import sys

logger = logging.getLogger("pillar_backbone")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
