"""Training configuration and metrics models."""

from typing import Union
from pathlib import Path
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    model_type: str = Field(default="yolov9", description="Model architecture")
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=64)
    device: str = Field(default="auto", description="Training device (auto, cpu, cuda)")
    workers: int = Field(default=8, ge=0)
    save_period: int = Field(default=10, ge=1, description="Save checkpoint every N epochs")
    patience: int = Field(default=50, ge=1, description="Early stopping patience")
    hyperparameters: dict[str, Union[str, Path]] = Field(default_factory=dict)


class TrainingMetrics(BaseModel):
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    val_loss: float | None = None
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map95: float | None = None
    learning_rate: float
    gpu_memory_mb: float | None = None
    time_seconds: float


class TrainingResult(BaseModel):
    """Complete training results."""
    config: TrainingConfig
    status: str = Field(..., description="completed, failed, stopped")
    best_epoch: int
    best_metrics: TrainingMetrics
    final_metrics: TrainingMetrics
    total_epochs: int
    total_time_seconds: float
    model_path: Path | None = None
    logs_path: Path | None = None
    error_message: str | None = None
    all_metrics: list[TrainingMetrics] = Field(default_factory=list)