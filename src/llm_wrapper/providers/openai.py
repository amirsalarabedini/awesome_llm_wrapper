"""OpenAI provider for LLM Wrapper."""

import json
import time
from typing import Dict, List, Any, Iterator, Union

import requests

from llm_wrapper.models import Message, Response, Provider
from llm_wrapper.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's API."""

    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, api_key: str):
        """Initialize the provider with an API key.

        Args:
            api_key: The API key for OpenAI.
        """
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def complete(self, prompt: str, **kwargs) -> Response:
        """Generate a completion for a prompt.

        Args:
            prompt: The prompt to generate a completion for.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            A Response object containing the completion.
        """
        # OpenAI doesn't have a dedicated completion endpoint anymore,
        # so we'll use the chat endpoint with a single user message
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: List[Union[Message, Dict[str, str]]], **kwargs) -> Response:
        """Generate a response for a conversation.

        Args:
            messages: A list of messages in the conversation.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            A Response object containing the response.
        """
        prepared_messages = self._prepare_messages(messages)
        model = kwargs.pop("model", self.DEFAULT_MODEL)
        
        payload = {
            "model": model,
            "messages": [msg.to_dict() for msg in prepared_messages],
            "max_tokens": kwargs.pop("max_tokens", 1000),
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()
            return Response.from_openai(data, model)
        except Exception as e:
            self._handle_error(e)

    def stream_chat(self, messages: List[Union[Message, Dict[str, str]]], **kwargs) -> Iterator[Response]:
        """Stream a response for a conversation.

        Args:
            messages: A list of messages in the conversation.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            An iterator of Response objects containing chunks of the response.
        """
        prepared_messages = self._prepare_messages(messages)
        model = kwargs.pop("model", self.DEFAULT_MODEL)
        
        payload = {
            "model": model,
            "messages": [msg.to_dict() for msg in prepared_messages],
            "stream": True,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove 'data: ' prefix
                if line.startswith(b"data: "):
                    line = line[6:]
                    
                # Skip [DONE] message
                if line == b"[DONE]":
                    break
                    
                try:
                    chunk = json.loads(line)
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        
                        # Only create a response if there's actual content
                        content = delta.get("content", "")
                        if content:
                            # Create a synthetic response with just this chunk
                            synthetic_response = {
                                "choices": [{
                                    "message": {"content": content},
                                    "finish_reason": chunk["choices"][0].get("finish_reason")
                                }]
                            }
                            yield Response.from_openai(synthetic_response, model)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Handle an error from the provider with OpenAI-specific handling.

        Args:
            error: The error to handle.

        Raises:
            ValueError: For configuration errors.
            requests.HTTPError: For HTTP errors, with improved context.
            Exception: For other errors, with improved context.
        """
        if isinstance(error, requests.HTTPError) and error.response is not None:
            status_code = error.response.status_code
            
            # Handle rate limiting with retries
            if status_code == 429:
                for retry in range(self.MAX_RETRIES):
                    try:
                        # Get retry delay from response headers or use default
                        retry_after = int(error.response.headers.get("Retry-After", self.RETRY_DELAY))
                        # Exponential backoff
                        wait_time = retry_after * (2 ** retry)
                        time.sleep(wait_time)
                        
                        # If we're still here, this is the last retry and it failed
                        if retry == self.MAX_RETRIES - 1:
                            # If we've exhausted all retries, raise the error
                            raise requests.HTTPError(
                                f"Rate limit exceeded after {self.MAX_RETRIES} retries: {error}"
                            )
                    except (ValueError, AttributeError):
                        # If we can't get a valid retry time, use exponential backoff
                        time.sleep(self.RETRY_DELAY * (2 ** retry))
        
        # Use the base error handling for other errors
        super()._handle_error(error)