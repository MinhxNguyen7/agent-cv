"""Data models for agent communication and configuration."""

from .dataset import (
    DatasetInfo as DatasetInfo,
)

from .training import (
    TrainingConfig as TrainingConfig,
    TrainingMetrics as TrainingMetrics,
    TrainingResult as TrainingResult,
)

from .messages import (
    Message as Message,
    TaskStatus as TaskStatus,
    TaskResult as TaskResult,
)
