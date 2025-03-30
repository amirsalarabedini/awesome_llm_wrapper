"""OpenAI provider for LLM Wrapper."""

import json
from typing import Dict, List, Any, Iterator, Union

import requests

from llm_wrapper.models import Message, Response, Provider
from llm_wrapper.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's API."""

    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

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
                        if "content" in delta and delta["content"]:
                            # Create a synthetic response with just this chunk
                            synthetic_response = {
                                "choices": [{
                                    "message": {"content": delta["content"]},
                                    "finish_reason": chunk["choices"][0].get("finish_reason")
                                }]
                            }
                            yield Response.from_openai(synthetic_response, model)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self._handle_error(e)