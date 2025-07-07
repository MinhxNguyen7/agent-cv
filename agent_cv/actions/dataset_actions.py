"""Dataset-related tool functions for agents."""

from pathlib import Path

from ..data_processing import DatasetAnalyzer
from ..models import DatasetInfo


def analyze_dataset(dataset_name: str) -> DatasetInfo:
    dataset_path = Path("datasets") / dataset_name
    analyzer = DatasetAnalyzer(dataset_path)
    return analyzer.analyze_dataset()
