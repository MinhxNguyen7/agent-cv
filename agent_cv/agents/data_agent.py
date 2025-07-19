from pathlib import Path

from agent_cv.models import Message


class DataAgent:
    """
    Utilizes VLMs to label data and write code to format it into YOLO format for training.
    """

    def __init__(self): ...

    def label_data(self, raw_dataset: Path, classes: list[str]) -> Path:
        """
        Uses VLMs to label the raw dataset.
        """
        ...

    def format_data(self, labeled_dataset: Path) -> Path:
        """
        Formats the labeled dataset into YOLO format.
        """
        ...
