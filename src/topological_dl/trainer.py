"""The implementation is split across smaller modules:

- ``pcn_backend``: PCX backend imports and shared dependencies.
- ``datasets``: dataset loading helpers.
- ``pcn_model``: model definition and energy functions.
- ``pcn_training``: batch training, evaluation, and optimizers.
- ``trainer_impl``: the ``Trainer`` class and analysis methods.

Existing imports such as ``from topological_dl.trainer import Trainer`` continue
to work through this module.
"""

try:
    from .data_loading import load_dataset
    from .pcn_model import Model, energy, forward
    from .pcn_training import eval, eval_on_batch, get_opts, train, train_on_batch
    from .trainer_impl import Trainer
except ImportError:
    from data_loading import load_dataset
    from pcn_model import Model, energy, forward
    from pcn_training import eval, eval_on_batch, get_opts, train, train_on_batch
    from trainer_impl import Trainer

__all__ = [
    "Trainer",
    "Model",
    "load_dataset",
    "forward",
    "energy",
    "train_on_batch",
    "eval_on_batch",
    "train",
    "eval",
    "get_opts",
]
