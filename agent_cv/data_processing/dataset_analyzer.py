"""Data loading and preprocessing utilities."""

import json
import random
from pathlib import Path

from ..models import DatasetConfig, DatasetInfo


class DatasetAnalyzer:
    """Handles dataset analysis, validation, and preprocessing."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate dataset configuration."""
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.config.data_path}")
        
        if not self.config.data_path.is_dir():
            raise ValueError(f"Data path must be a directory: {self.config.data_path}")
    
    def analyze_dataset(self) -> DatasetInfo:
        """Analyze the dataset structure and metadata."""
        image_files = self._find_image_files()
        annotation_files = self._find_annotation_files()
        
        # Validate image-annotation pairs
        valid_pairs = self._validate_pairs(image_files, annotation_files)
        
        # Analyze class distribution
        class_distribution = self._analyze_classes(valid_pairs)
        
        # Create data splits
        train_samples, val_samples, test_samples = self._create_splits(valid_pairs)
        
        return DatasetInfo(
            config=self.config,
            total_images=len(image_files),
            total_annotations=len(annotation_files),
            class_distribution=class_distribution,
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            test_samples=len(test_samples),
        )
    
    def _find_image_files(self) -> list[Path]:
        """Find all image files in the dataset directory."""
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(self.config.data_path.rglob(f"*{ext}"))
        return sorted(image_files)
    
    def _find_annotation_files(self) -> list[Path]:
        """Find all annotation files based on format."""
        if self.config.annotation_format == "yolo":
            return sorted(self.config.data_path.rglob("*.txt"))
        elif self.config.annotation_format == "coco":
            return sorted(self.config.data_path.rglob("*.json"))
        else:
            raise ValueError(f"Unsupported annotation format: {self.config.annotation_format}")
    
    def _validate_pairs(self, image_files: list[Path], annotation_files: list[Path]) -> list[tuple[Path, Path]]:
        """Validate that each image has a corresponding annotation file."""
        valid_pairs = []
        
        # Create lookup for annotation files
        annotation_lookup = {f.stem: f for f in annotation_files}
        
        for image_file in image_files:
            if image_file.stem in annotation_lookup:
                valid_pairs.append((image_file, annotation_lookup[image_file.stem]))
        
        return valid_pairs
    
    def _analyze_classes(self, valid_pairs: list[tuple[Path, Path]]) -> dict[str, int]:
        """Analyze class distribution in the dataset."""
        class_counts = {}
        
        for _, annotation_file in valid_pairs:
            if self.config.annotation_format == "yolo":
                classes = self._parse_yolo_classes(annotation_file)
            else:
                classes = self._parse_coco_classes(annotation_file)
            
            for class_id in classes:
                class_name = self._get_class_name(class_id)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts
    
    def _parse_yolo_classes(self, annotation_file: Path) -> list[int]:
        """Parse YOLO format annotation file to extract class IDs."""
        classes = []
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.append(int(parts[0]))
        except (ValueError, IndexError):
            pass  # Skip malformed lines
        
        return classes
    
    def _parse_coco_classes(self, annotation_file: Path) -> list[int]:
        """Parse COCO format annotation file to extract class IDs."""
        classes = []
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                
            # Handle both COCO annotation format and single image annotation
            if 'annotations' in data:
                # Full COCO format
                for annotation in data['annotations']:
                    if 'category_id' in annotation:
                        classes.append(annotation['category_id'])
            elif 'category_id' in data:
                # Single annotation format
                classes.append(data['category_id'])
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass  # Skip malformed files
            
        return classes
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        if self.config.class_names and class_id < len(self.config.class_names):
            return self.config.class_names[class_id]
        return f"class_{class_id}"
    
    def _create_splits(self, valid_pairs: list[tuple[Path, Path]]) -> tuple[list, list, list]:
        """Create train/validation/test splits."""
        # Shuffle pairs for random splitting
        shuffled_pairs = valid_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        total = len(shuffled_pairs)
        train_end = int(total * self.config.split.train_ratio)
        val_end = train_end + int(total * self.config.split.val_ratio)
        
        train_samples = shuffled_pairs[:train_end]
        val_samples = shuffled_pairs[train_end:val_end]
        test_samples = shuffled_pairs[val_end:]
        
        return train_samples, val_samples, test_samples