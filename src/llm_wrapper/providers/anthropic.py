"""Anthropic provider for LLM Wrapper."""

import json
from typing import Dict, List, Any, Iterator, Union

import requests

from llm_wrapper.models import Message, Response, Provider
from llm_wrapper.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude API."""

    BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-3-5-sonnet"  

    def __init__(self, api_key: str):
        """Initialize the provider with an API key.

        Args:
            api_key: The API key for Anthropic.
        """
        super().__init__(api_key)
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
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
        # Anthropic uses the same endpoint for completions and chat
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
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in prepared_messages:
            if msg.role == "system":
                # Add system message as a user message with system prefix
                anthropic_messages.append({
                    "role": "user",
                    "content": f"<system>\n{msg.content}\n</system>"
                })
            else:
                # Handle standard user and assistant messages
                role = "user" if msg.role == "user" else "assistant"
                anthropic_messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 1000),
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/messages",
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()
            return Response.from_anthropic(data, model)
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
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in prepared_messages:
            if msg.role == "system":
                # Add system message as a user message with system prefix
                anthropic_messages.append({
                    "role": "user",
                    "content": f"<system>\n{msg.content}\n</system>"
                })
            else:
                # Handle standard user and assistant messages
                role = "user" if msg.role == "user" else "assistant"
                anthropic_messages.append({
                    "role": role,
                    "content": msg.content
                })
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 1000),
            "stream": True,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/messages",
                headers=self.headers,
                data=json.dumps(payload),
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line or line.strip() == b"":
                    continue
                    
                if line.startswith(b"data: "):
                    line = line[6:]
                    
                if line == b"[DONE]":
                    break
                    
                try:
                    chunk = json.loads(line)
                    # Handle different types of events in the stream
                    if "type" in chunk:
                        event_type = chunk.get("type")
                        
                        # Content block contains text
                        if event_type == "content_block_delta" and "delta" in chunk:
                            delta = chunk["delta"]
                            if "text" in delta:
                                synthetic_response = {
                                    "content": [{"text": delta["text"]}],
                                    "stop_reason": None
                                }
                                yield Response.from_anthropic(synthetic_response, model)
                        
                        # Message is complete
                        elif event_type == "message_stop":
                            synthetic_response = {
                                "content": [{"text": ""}],
                                "stop_reason": chunk.get("stop_reason")
                            }
                            yield Response.from_anthropic(synthetic_response, model)
                    
                    # Fallback for older API versions or different formats
                    elif "content" in chunk and chunk["content"]:
                        synthetic_response = {
                            "content": [{"text": chunk["content"]}],
                            "stop_reason": chunk.get("stop_reason")
                        }
                        yield Response.from_anthropic(synthetic_response, model)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self._handle_error(e)