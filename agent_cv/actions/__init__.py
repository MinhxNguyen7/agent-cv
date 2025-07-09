"""
Agent-callable action functions.

The code will be from code blocks returned by the agents, which
will then be executed in a notebook environment
"""

from .dataset_actions import make_dataset_analyzer as make_dataset_analyzer
from .training_actions import train_model as train_model


ACTIONS = (
    make_dataset_analyzer,
    train_model,
)
