"""Dataset-related tool functions for agents."""

from pathlib import Path

from ..data_processing import DatasetAnalyzer


DATASET_DIRECTORY = Path("datasets")


def make_dataset_analyzer(dataset_name: str) -> DatasetAnalyzer:
    """
    Creates a DatasetAnalyzer instance for the specified dataset.

    The analyzer can be used as follows:

    ```python
    from agent_cv.actions import get_dataset_analyzer
    analyzer = get_dataset_analyzer("path/to/dataset")

    # Returns printable JSON-like
    dataset_info = analyzer.analyze_dataset()

    # Get n sample images from each split of the dataset
    sample_images = analyzer.get_sample_images(n=2)
    ```
    """
    dataset_path = DATASET_DIRECTORY / dataset_name

    if not dataset_path.exists():
        raise ValueError(f"Dataset {dataset_name} does not exist at {dataset_path}")

    return DatasetAnalyzer(dataset_path)
