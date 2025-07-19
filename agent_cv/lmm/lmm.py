from abc import ABC, abstractmethod
from typing import Iterable
import base64
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam

from ..models import Message


class LMM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(self, chat: Iterable[Message]) -> str:
        """
        Call the language model with a chat history.

        Args:
            chat (Iterable[AgentMessage]): The chat history to send to the model.

        Returns:
            str: The model's response.
        """
        return NotImplemented


class AnthropicLMM(LMM):
    def __init__(
        self, model_name: str = "claude-3-5-sonnet-20241022", api_key: str | None = None
    ):
        super().__init__(model_name)
        self.client = Anthropic(api_key=api_key)

    def __call__(self, chat: Iterable[Message]) -> str:
        """
        Call the Anthropic language model with a chat history.

        Args:
            chat (Iterable[AgentMessage]): The chat history to send to the model.

        Returns:
            str: The model's response.
        """
        messages = [
            self._convert_to_anthropic_format(msg)
            for msg in chat
            if isinstance(msg, Message)
        ]

        # Ensure we have at least one message and it starts with user
        if not messages:
            raise ValueError("No messages provided to the model")

        # If the first message is from assistant, prepend a user message
        if messages[0]["role"] == "assistant":
            messages.insert(
                0,
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Continue the conversation."}],
                },
            )

        # Call the Anthropic API
        response = self.client.messages.create(
            model=self.model_name, max_tokens=4096, messages=messages
        )

        return response.content[0].text  # type: ignore

    def _convert_to_anthropic_format(self, msg: Message) -> MessageParam:
        """
        Convert an AgentMessage to Anthropic API format.

        Args:
            msg: The AgentMessage to convert

        Returns:
            MessageParam formatted for Anthropic API
        """
        content_blocks = []

        if isinstance(msg.content, dict):
            for key, value in msg.content.items():
                if isinstance(value, Path):
                    # Handle media files
                    media_type = self._get_media_type(value)
                    if media_type:
                        # Read and encode the media file
                        with open(value, "rb") as f:
                            media_data = base64.b64encode(f.read()).decode("utf-8")

                        content_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": media_data,
                                },
                            }
                        )
                    else:
                        # Handle non-media files as text
                        content_blocks.append(
                            {"type": "text", "text": f"{key}: {value}"}
                        )
                else:
                    # Handle text content
                    content_blocks.append({"type": "text", "text": f"{key}: {value}"})
        else:
            # Handle simple string content
            content_blocks.append({"type": "text", "text": str(msg.content)})

        role = "user" if msg.sender == "User" else "assistant"
        return {"role": role, "content": content_blocks}

    def _get_media_type(self, file_path: Path) -> str | None:
        """
        Get the media type for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Media type string or None if not a supported media file
        """
        suffix = file_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix)
