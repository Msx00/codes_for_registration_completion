"""Thin wrapper for ShapeNet34 training.

This repo's training loop in `core/train_55.py` actually operates on complete
point clouds (gtcloud) and generates partial clouds online, which matches the
ShapeNet34 split as well.

Keeping it as a wrapper avoids duplicating a large training script.
"""

from core.train_55 import train_net  # re-export

__all__ = ["train_net"]
