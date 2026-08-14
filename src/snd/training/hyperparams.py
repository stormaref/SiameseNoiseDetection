"""The model/optimization hyperparameters shared by the cleaner, detector and trainer.

Before this existed, every one of these values was declared three times -- once on
``NoiseCleaner.__init__``, once on ``NoiseDetector.__init__``, and once again at the
call site that forwarded 26 keyword arguments between them. Adding a hyperparameter
meant editing all three. They now travel as a single :class:`TrainingConfig`.

The per-dataset dicts in :mod:`snd.config` still use the flat keyword form, so
``NoiseCleaner(dataset, **CIFAR10_30_PARAMS)`` keeps working; the cleaner splits
those kwargs into this config plus its own orchestration settings.
"""
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class TrainingConfig:
    """Everything needed to build and train one Siamese ensemble member."""

    # architecture
    model: str = 'resnet18'
    embedding_dimension: int = 128
    pre_trained: bool = True
    trainable: bool = True
    dropout_prob: float = 0.5
    cnn_size: int | None = None
    siamese_middle_size: int | None = None
    parallel: bool = False
    num_classes: int = 10

    # optimization
    optimizer: str = 'Adam'
    weight_decay: float = 0.001
    patience: int = 5
    batch_size: int = 256

    # objective
    loss: str = 'ce'
    label_smoothing: float = 0.1
    contrastive_ratio: float = 3
    distance_meter: str = 'euclidian'
    margin: float = 5
    freeze_epoch: int | None = 10

    # pair sampling
    train_pairs: int = 12000
    val_pairs: int = 5000

    # data plumbing
    transform: Any = None
    augmented_transform: Any = None

    @classmethod
    def field_names(cls) -> set[str]:
        """Names this config accepts, for splitting flat keyword arguments."""
        return {f.name for f in fields(cls)}

    @classmethod
    def split_kwargs(cls, kwargs: dict) -> tuple['TrainingConfig', dict]:
        """Partition flat kwargs into (TrainingConfig, everything else).

        Accepts the legacy aliases used by the per-dataset config dicts:
        ``num_class`` -> ``num_classes`` and ``training_batch_size`` -> ``batch_size``.
        """
        kwargs = dict(kwargs)
        for legacy, current in (('num_class', 'num_classes'),
                                ('training_batch_size', 'batch_size')):
            if legacy in kwargs:
                kwargs.setdefault(current, kwargs.pop(legacy))
        names = cls.field_names()
        mine = {k: kwargs.pop(k) for k in list(kwargs) if k in names}
        return cls(**mine), kwargs

    def replace(self, **changes) -> 'TrainingConfig':
        """Return a copy with `changes` applied."""
        merged = {f.name: getattr(self, f.name) for f in fields(self)}
        merged.update(changes)
        return TrainingConfig(**merged)
