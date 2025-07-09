"""Data loading and preprocessing utilities."""

from itertools import chain
import json
import random
from pathlib import Path
from typing import Any, NamedTuple, Optional
from collections import defaultdict, Counter
import logging

from ..models import DatasetInfo


class ImagePaths(NamedTuple):
    train: list[Path]
    val: list[Path]
    test: list[Path]


class DatasetAnalyzer:
    """
    Analyzes the structure and metadata of a dataset.

    This class supports the YOLO dataset format only.
    """

    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    def __init__(self, dataset_path: Path | str):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        self._validate_path()
        self._images: ImagePaths | None = None
        self._annotations: list[Path] = []
        self._class_names: Optional[list[str]] = None
        self._format: Optional[str] = None

    def _validate_path(self) -> None:
        """Validate dataset path."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path must be a directory: {self.dataset_path}")

    def _discover_images(self) -> ImagePaths:
        """Discover all image files in the dataset."""
        return ImagePaths(
            train=list(
                chain(
                    *(
                        self.dataset_path.rglob(f"*{ext}")
                        for ext in self.IMAGE_EXTENSIONS
                    )
                )
            ),
            val=list(
                chain(
                    *(
                        self.dataset_path.rglob(f"*{ext}")
                        for ext in self.IMAGE_EXTENSIONS
                    )
                )
            ),
            test=list(
                chain(
                    *(
                        self.dataset_path.rglob(f"*{ext}")
                        for ext in self.IMAGE_EXTENSIONS
                    )
                )
            ),
        )

    def _discover_annotations(self) -> list[Path]:
        """Discover annotation files in the dataset."""
        annotations = []
        annotations.extend(self.dataset_path.rglob("*.txt"))  # YOLO format
        annotations.extend(self.dataset_path.rglob("*.json"))  # COCO format
        annotations.extend(self.dataset_path.rglob("*.xml"))  # Pascal VOC format
        return sorted(annotations)

    def _detect_format(self) -> str:
        """Detect the annotation format of the dataset."""
        if not self._annotations:
            return "unknown"

        # Check for YOLO format (txt files with normalized coordinates)
        txt_files = [f for f in self._annotations if f.suffix == ".txt"]
        if txt_files:
            try:
                with open(txt_files[0], "r") as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5 and all(
                            0 <= float(p) <= 1 for p in parts[1:5]
                        ):
                            return "yolo"
            except (ValueError, IndexError):
                pass

        # Check for COCO format (JSON files)
        json_files = [f for f in self._annotations if f.suffix == ".json"]
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    data = json.load(f)
                    if "annotations" in data and "images" in data:
                        return "coco"
            except (json.JSONDecodeError, KeyError):
                pass

        # Check for Pascal VOC format (XML files)
        xml_files = [f for f in self._annotations if f.suffix == ".xml"]
        if xml_files:
            return "pascal_voc"

        return "unknown"

    def _parse_yolo_annotations(self) -> tuple[list[str], dict[str, int]]:
        """Parse YOLO format annotations."""
        class_counter = Counter()
        class_names = []

        # Look for classes.txt or similar
        classes_file = self.dataset_path / "classes.txt"
        if not classes_file.exists():
            classes_file = self.dataset_path / "obj.names"

        if classes_file.exists():
            with open(classes_file, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

        # Count classes from annotation files
        for ann_file in self._annotations:
            if ann_file.suffix == ".txt":
                try:
                    with open(ann_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                class_counter[class_id] += 1
                except (ValueError, IndexError):
                    continue

        # Create class names if not found
        if not class_names:
            max_class_id = max(class_counter.keys()) if class_counter else 0
            class_names = [f"class_{i}" for i in range(max_class_id + 1)]

        # Convert to string-based counter
        class_distribution = {}
        for class_id, count in class_counter.items():
            if class_id < len(class_names):
                class_distribution[class_names[class_id]] = count

        return class_names, class_distribution

    def _parse_coco_annotations(self) -> tuple[list[str], dict[str, int]]:
        """Parse COCO format annotations."""
        class_names = []
        class_distribution = defaultdict(int)

        for ann_file in self._annotations:
            if ann_file.suffix == ".json":
                try:
                    with open(ann_file, "r") as f:
                        data = json.load(f)

                        # Get categories
                        categories = data.get("categories", [])
                        if categories and not class_names:
                            class_names = [cat["name"] for cat in categories]

                        # Count annotations
                        for ann in data.get("annotations", []):
                            cat_id = ann["category_id"]
                            if cat_id < len(class_names):
                                class_distribution[class_names[cat_id]] += 1

                except (json.JSONDecodeError, KeyError):
                    continue

        return class_names, dict(class_distribution)

    def analyze_dataset(self) -> DatasetInfo:
        """Analyze the dataset and return comprehensive information."""
        self.logger.info(f"Analyzing dataset at {self.dataset_path}")

        # Discover files
        self._images = self._discover_images()
        self._annotations = self._discover_annotations()

        if not self._images:
            raise ValueError("No images found in dataset")

        # Detect format and parse annotations
        self._format = self._detect_format()

        class_names = []
        class_distribution = {}

        if self._format == "yolo":
            class_names, class_distribution = self._parse_yolo_annotations()
        elif self._format == "coco":
            class_names, class_distribution = self._parse_coco_annotations()

        # Calculate split counts
        total_images = len(self._images)
        split = (
            len(self._images.train) / total_images if total_images > 0 else 0,
            len(self._images.val) / total_images if total_images > 0 else 0,
            len(self._images.test) / total_images if total_images > 0 else 0,
        )

        # Create dataset info
        dataset_info = DatasetInfo(
            name=self.dataset_path.name,
            path=self.dataset_path,
            class_names=class_names if class_names else None,
            class_distribution=class_distribution,
            total_images=total_images,
            split=split,
            train_samples=len(self._images.train),
            val_samples=len(self._images.val),
            test_samples=len(self._images.test),
        )

        self.logger.info(
            f"Dataset analysis complete: {total_images} images, {len(class_names)} classes"
        )

        return dataset_info

    def get_sample_images(self, n: int = 5) -> ImagePaths:
        """
        Get `n` random images from each split of the dataset.

        Returns a 3-tuple of lists containing random images from train, val, and test splits.
        """
        if not self._images:
            self._images = self._discover_images()

        return ImagePaths(
            train=random.sample(self._images.train, min(n, len(self._images.train))),
            val=random.sample(self._images.val, min(n, len(self._images.val))),
            test=random.sample(self._images.test, min(n, len(self._images.test))),
        )
