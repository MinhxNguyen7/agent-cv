"""Agent communication and task management models."""

from typing import Literal
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field


role_t = Literal[
    "User", "CodeOutput", "OrchestratorAgent", "TrainingAgent", "DeploymentAgent"
]


class TaskStatus(Enum):
    """Status of a task or operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(BaseModel):
    sender: role_t = Field(..., description="Type of sender agent")
    recipient: role_t = Field(..., description="Type of recipient agent")
    content: dict[str, str | Path] = Field(default_factory=dict)
    timestamp: float | None = None
    correlation_id: str | None = None


class TaskResult(BaseModel):
    """Result of a completed task."""

    task_id: str
    status: TaskStatus
    result: dict[str, str | Path] | None = None
    error_message: str | None = None
    metadata: dict[str, str | Path] = Field(default_factory=dict)
    duration_seconds: float | None = None

    def __str__(self) -> str:
        return str(self.model_dump())
