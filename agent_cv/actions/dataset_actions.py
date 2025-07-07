"""Dataset-related tool functions for agents."""

from pathlib import Path

from ..data_processing import DatasetAnalyzer
from ..models import DatasetConfig, DatasetInfo


def analyze_dataset(data_path: Path, **kwargs) -> DatasetInfo:
    """Analyze a dataset structure and metadata.

    Args:
        data_path: Path to dataset directory
        **kwargs: Additional configuration options

    Returns:
        DatasetInfo with dataset metadata and statistics
    """
    config = DatasetConfig(
        name=kwargs.get("name", data_path.name), data_path=data_path, **kwargs
    )

    analyzer = DatasetAnalyzer(config)
    return analyzer.analyze_dataset()
