"""Training agent for automated model training and optimization."""

import time
import logging
from pathlib import Path

from ..lmm import AnthropicLMM
from ..models import Message, TrainingConfig, TrainingResult, DatasetInfo


logger = logging.getLogger(__name__)


class TrainingAgent:
    """
    Agent responsible for model training decision-making and optimization.

    This agent handles:
    - Training configuration optimization
    - Model training execution
    - Training strategy adaptation

    It has access to a notebook to execute code (actions),
    which it can use to analyze/inspect the data, train models, and evaluate results.
    """

    def __init__(self, dataset_info: DatasetInfo, lmm: AnthropicLMM | None = None):
        self.dataset_info = dataset_info
        self.lmm = lmm or AnthropicLMM()

    def perform_training(
        self, max_messages: int = 20
    ) -> tuple[TrainingResult, list[Message]]:
        """
        Perform the training process based on the dataset information by
        writing code to a notebook and executing it.

        Args:
            max_messages (int): Maximum number of messages to exchange with the LMM
            before it must stop.

        Returns:
            tuple[TrainingResult, list[AgentMessage]]: The training result and messages exchanged.
        """
        ...
