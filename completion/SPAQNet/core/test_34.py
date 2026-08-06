"""Thin wrapper for ShapeNet34 testing.

Reuses the evaluation logic from `core/test_55.py`.
"""

from core.test_55 import test_net  # re-export

__all__ = ["test_net"]
