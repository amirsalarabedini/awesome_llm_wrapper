"""Base provider class for LLM Wrapper."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator, Union

from llm_wrapper.models import Message, Response


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, api_key: str):
        """Initialize the provider with an API key.

        Args:
            api_key: The API key for the provider.
        """
        self.api_key = api_key

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> Response:
        """Generate a completion for a prompt.

        Args:
            prompt: The prompt to generate a completion for.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            A Response object containing the completion.
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Union[Message, Dict[str, str]]], **kwargs) -> Response:
        """Generate a response for a conversation.

        Args:
            messages: A list of messages in the conversation.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            A Response object containing the response.
        """
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Union[Message, Dict[str, str]]], **kwargs) -> Iterator[Response]:
        """Stream a response for a conversation.

        Args:
            messages: A list of messages in the conversation.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            An iterator of Response objects containing chunks of the response.
        """
        pass

    def _prepare_messages(self, messages: List[Union[Message, Dict[str, str]]]) -> List[Message]:
        """Convert messages to Message objects if they are dictionaries.

        Args:
            messages: A list of messages in the conversation.

        Returns:
            A list of Message objects.
        """
        result = []
        for message in messages:
            if isinstance(message, dict):
                result.append(Message.from_dict(message))
            else:
                result.append(message)
        return result

    def _handle_error(self, error: Exception) -> None:
        """Handle an error from the provider.

        Args:
            error: The error to handle.

        Raises:
            The error, possibly with additional context.
        """
        # This can be overridden by subclasses to provide provider-specific error handling
        raise error