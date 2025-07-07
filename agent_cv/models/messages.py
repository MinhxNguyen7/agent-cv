"""Agent communication and task management models."""

from typing import Literal, Union
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field


agent_type = Literal["User", "Orchestrator", "Training", "Evaluation", "Deployment"]


class TaskStatus(str, Enum):
    """Status of a task or operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentMessage(BaseModel):
    """Message between agents."""

    sender: agent_type = Field(..., description="Type of sender agent")
    recipient: agent_type = Field(..., description="Type of recipient agent")
    message_type: str = Field(..., description="Type of message")
    content: dict[str, Union[str, Path]] = Field(default_factory=dict)
    timestamp: float | None = None
    correlation_id: str | None = None


class TaskResult(BaseModel):
    """Result of a completed task."""

    task_id: str
    status: TaskStatus
    result: dict[str, Union[str, Path]] | None = None
    error_message: str | None = None
    metadata: dict[str, Union[str, Path]] = Field(default_factory=dict)
    duration_seconds: float | None = None
