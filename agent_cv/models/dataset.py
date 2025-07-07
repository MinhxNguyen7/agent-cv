"""Dataset configuration and metadata models."""

from pathlib import Path
from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""

    name: str = Field(..., description="Dataset name")
    path: Path = Field(..., description="Path to dataset directory")

    class_names: list[str] | None = Field(
        default=None, description="List of class names"
    )
    class_distribution: dict[str, int]

    total_images: int
    split: tuple[float, float, float] = Field(
        default=(0.8, 0.1, 0.1),
        description="Train/val/test split ratios (sum should be 1.0)",
    )

    train_samples: int
    val_samples: int
    test_samples: int
