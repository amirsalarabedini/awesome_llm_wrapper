"""Base provider class for LLM Wrapper."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator, Union
import requests
import json

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
            
        Raises:
            ValueError: If a message dictionary is missing required fields.
        """
        result = []
        for message in messages:
            if isinstance(message, dict):
                # Validate required fields
                if "role" not in message:
                    raise ValueError("Message dictionary missing required 'role' field")
                if "content" not in message:
                    raise ValueError("Message dictionary missing required 'content' field")
                    
                result.append(Message.from_dict(message))
            elif isinstance(message, Message):
                result.append(message)
            else:
                raise ValueError(f"Message must be a Message object or dictionary, got {type(message)}")
        return result

    def _handle_error(self, error: Exception) -> None:
        """Handle an error from the provider.

        Args:
            error: The error to handle.

        Raises:
            ValueError: For configuration errors.
            requests.HTTPError: For HTTP errors, with improved context.
            Exception: For other errors, with improved context.
        """
        if isinstance(error, requests.HTTPError):
            # Extract response data if available
            try:
                response = error.response
                status_code = response.status_code
                error_data = response.json()
                if 400 <= status_code < 500:
                    if status_code == 401:
                        raise ValueError(f"Authentication error: Invalid API key or unauthorized access (Status: {status_code})")
                    elif status_code == 429:
                        raise requests.HTTPError(f"Rate limit exceeded: {error_data.get('error', {}).get('message', 'Too many requests')} (Status: {status_code})")
                    else:
                        raise requests.HTTPError(f"API request error: {error_data.get('error', {}).get('message', str(error))} (Status: {status_code})")
                else:
                    raise requests.HTTPError(f"Server error: {error_data.get('error', {}).get('message', str(error))} (Status: {status_code})")
            except (ValueError, AttributeError):
                # If we can't extract structured data, just add context to the original error
                raise requests.HTTPError(f"HTTP Error: {str(error)}")
        elif isinstance(error, (json.JSONDecodeError, TypeError, KeyError)):
            raise ValueError(f"Error processing response: {str(error)}") 
        else:
            # Re-raise other exceptions with better context
            raise Exception(f"Provider API error: {str(error)}")