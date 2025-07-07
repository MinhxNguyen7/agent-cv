"""Dataset configuration and metadata models."""

from pathlib import Path
from typing import Union
from pydantic import BaseModel, Field


class DataSplit(BaseModel):
    """Configuration for train/validation/test splits."""

    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        """Validate that ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and processing."""

    name: str = Field(..., description="Dataset name")
    data_path: Path = Field(..., description="Path to dataset directory")
    image_extensions: list[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp"])
    annotation_format: str = Field(
        default="yolo", description="Annotation format (yolo, coco, etc.)"
    )
    class_names: list[str] | None = Field(
        default=None, description="List of class names"
    )
    split: DataSplit = Field(default_factory=DataSplit)
    preprocessing: dict[str, Union[str, Path]] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""

    config: DatasetConfig
    total_images: int
    total_annotations: int
    class_distribution: dict[str, int]
    train_samples: int
    val_samples: int
    test_samples: int
