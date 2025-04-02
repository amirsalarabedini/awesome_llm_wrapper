"""Gemini provider for LLM Wrapper."""

import json
from typing import Dict, List, Any, Iterator, Union

import requests

from llm_wrapper.models import Message, Response, Provider
from llm_wrapper.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider for Google's Gemini API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1"
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str):
        """Initialize the provider with an API key.

        Args:
            api_key: The API key for Google Gemini.
        """
        super().__init__(api_key)
        self.api_key = api_key

    def complete(self, prompt: str, **kwargs) -> Response:
        """Generate a completion for a prompt.

        Args:
            prompt: The prompt to generate a completion for.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            A Response object containing the completion.
        """
        # Gemini uses the same endpoint for completions and chat
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
        
        # Convert messages to Gemini format
        gemini_messages = []
        system_content = None
        
        # Check for system message first
        for msg in prepared_messages:
            if msg.role == "system":
                system_content = msg.content
                break
                
        for msg in prepared_messages:
            if msg.role == "system":
                # System message is handled separately, skip here
                continue
                
            role = "user" if msg.role == "user" else "model"
            
            # Add system instruction to the first user message if available
            if role == "user" and system_content and not gemini_messages:
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": f"System instruction: {system_content}\n\nUser message: {msg.content}"}]
                })
            else:
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        payload = {
            "contents": gemini_messages
        }
        
        # Add max_tokens parameter with the correct name for Gemini API
        if "max_tokens" in kwargs:
            payload["maxOutputTokens"] = kwargs.pop("max_tokens")
        
        # Convert snake_case to camelCase for Gemini API parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs.pop("temperature")
            
        if "top_p" in kwargs:
            payload["topP"] = kwargs.pop("top_p")
            
        if "top_k" in kwargs:
            payload["topK"] = kwargs.pop("top_k")
            
        # Add any remaining kwargs
        payload.update(kwargs)

        try:
            url = f"{self.BASE_URL}/models/{model}:generateContent?key={self.api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()
            return Response.from_gemini(data, model)
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
        
        # Convert messages to Gemini format
        gemini_messages = []
        system_content = None
        
        # Check for system message first
        for msg in prepared_messages:
            if msg.role == "system":
                system_content = msg.content
                break
                
        for msg in prepared_messages:
            if msg.role == "system":
                # System message is handled separately, skip here
                continue
                
            role = "user" if msg.role == "user" else "model"
            
            # Add system instruction to the first user message if available
            if role == "user" and system_content and not gemini_messages:
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": f"System instruction: {system_content}\n\nUser message: {msg.content}"}]
                })
            else:
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        payload = {
            "contents": gemini_messages,
            "stream": True
        }
        
        # Add max_tokens parameter with the correct name for Gemini API
        if "max_tokens" in kwargs:
            payload["maxOutputTokens"] = kwargs.pop("max_tokens")
            
        # Convert snake_case to camelCase for Gemini API parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs.pop("temperature")
            
        if "top_p" in kwargs:
            payload["topP"] = kwargs.pop("top_p")
            
        if "top_k" in kwargs:
            payload["topK"] = kwargs.pop("top_k")
            
        # Add any remaining kwargs
        payload.update(kwargs)

        try:
            url = f"{self.BASE_URL}/models/{model}:streamGenerateContent?key={self.api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line or line.strip() == b"":
                    continue
                    
                try:
                    chunk = json.loads(line)
                    if "candidates" in chunk and chunk["candidates"]:
                        content = chunk["candidates"][0].get("content", {})
                        parts = content.get("parts", [])
                        if parts and "text" in parts[0]:
                            yield Response.from_gemini(chunk, model)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self._handle_error(e)