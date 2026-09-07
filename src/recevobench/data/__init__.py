"""Dataset preparation adapters and the normalized interaction format."""

from .base import DatasetCard, Interaction, PreparationError, verify_dataset_card
from .registry import get_adapter, prepare_task

__all__ = [
    "DatasetCard",
    "Interaction",
    "PreparationError",
    "get_adapter",
    "prepare_task",
    "verify_dataset_card",
]
