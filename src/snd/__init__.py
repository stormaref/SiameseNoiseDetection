"""Instance-dependent label noise correction via contrastive learning and ensemble disagreement.

Sub-packages are imported lazily -- importing :mod:`snd` itself is cheap and has
no side effects. Note that :mod:`snd.config` instantiates the torchvision
datasets at import time (downloading them into ``data/`` if missing).
"""

__version__ = "0.2.0"

__all__ = ["data", "models", "training", "pipeline", "evaluation"]
