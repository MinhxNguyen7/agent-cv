"""Dataset-related tool functions for agents."""

from pathlib import Path

from ..data_processing import DatasetAnalyzer
from ..models import DatasetInfo


def analyze_dataset(dataset_name: str) -> DatasetInfo:
    """Analyze a dataset and return comprehensive information about it.
    
    This function analyzes a dataset located in the datasets directory and returns
    detailed information including class distribution, image counts, and dataset structure.
    The results are printed to stdout for agent visibility.
    
    Args:
        dataset_name: Name of the dataset directory in datasets/
        
    Returns:
        DatasetInfo: Comprehensive dataset information including:
            - Dataset path and name
            - Class distribution and counts
            - Train/val/test split information
            - Image format and size statistics
            
    Prints:
        Complete dataset analysis results to stdout for agent consumption
    """
    print(f"🔍 Analyzing dataset: {dataset_name}")
    
    dataset_path = Path("datasets") / dataset_name
    analyzer = DatasetAnalyzer(dataset_path)
    result = analyzer.analyze_dataset()
    
    print(f"✅ Dataset analysis complete:")
    print(result)
    
    return result
