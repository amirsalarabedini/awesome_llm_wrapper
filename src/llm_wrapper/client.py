"""Client for the LLM Wrapper."""

import os
from typing import Dict, List, Optional, Union, Iterator

from llm_wrapper.models import Provider, Message, Response
from llm_wrapper.providers.base import BaseProvider


class LLMClient:
    """Client for interacting with multiple LLM providers."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None
    ):
        """Initialize the client with API keys for different providers.

        Args:
            openai_api_key: API key for OpenAI.
            gemini_api_key: API key for Google Gemini.
            anthropic_api_key: API key for Anthropic.
            
        Raises:
            ValueError: If all API keys are None or empty strings.
        """
        self.providers = {}

        # Initialize providers with provided API keys
        if openai_api_key and openai_api_key.strip():
            from llm_wrapper.providers.openai import OpenAIProvider
            self.providers[Provider.OPENAI] = OpenAIProvider(openai_api_key)

        if gemini_api_key and gemini_api_key.strip():
            from llm_wrapper.providers.gemini import GeminiProvider
            self.providers[Provider.GEMINI] = GeminiProvider(gemini_api_key)

        if anthropic_api_key and anthropic_api_key.strip():
            from llm_wrapper.providers.anthropic import AnthropicProvider
            self.providers[Provider.ANTHROPIC] = AnthropicProvider(anthropic_api_key)
            
        # Validate that at least one provider is initialized
        if not self.providers:
            raise ValueError(
                "No valid API keys provided. At least one provider API key is required."
            )

    @classmethod
    def from_env(cls) -> 'LLMClient':
        """Initialize the client with API keys from environment variables.

        Returns:
            An initialized LLMClient.
            
        Raises:
            ValueError: If no valid API keys found in environment variables.
        """
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        
        if not any([openai_key.strip(), gemini_key.strip(), anthropic_key.strip()]):
            raise ValueError(
                "No valid API keys found in environment variables. "
                "Set at least one of OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY."
            )
            
        return cls(
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
            anthropic_api_key=anthropic_key
        )

    def _get_provider(self, provider_name: Union[str, Provider]) -> BaseProvider:
        """Get a provider by name.

        Args:
            provider_name: The name of the provider.

        Returns:
            The provider.

        Raises:
            ValueError: If the provider is not supported or not initialized.
        """
        if isinstance(provider_name, str):
            try:
                provider_name = Provider(provider_name.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider_name}")

        if provider_name not in self.providers:
            raise ValueError(
                f"Provider {provider_name} not initialized. "
                f"Please provide an API key for this provider."
            )

        return self.providers[provider_name]

    def complete(self, provider: Union[str, Provider], prompt: str, model: Optional[str] = None, **kwargs) -> Response:
        """Generate a completion for a prompt.

        Args:
            provider: The provider to use.
            prompt: The prompt to generate a completion for.
            model: The specific model to use (optional). If not provided, the provider's default will be used.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            A Response object containing the completion.
        """
        provider_instance = self._get_provider(provider)
        if model:
            kwargs['model'] = model
        return provider_instance.complete(prompt, **kwargs)

    def chat(
        self,
        provider: Union[str, Provider],
        messages: List[Union[Message, Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs
    ) -> Response:
        """Generate a response for a conversation.

        Args:
            provider: The provider to use.
            messages: A list of messages in the conversation.
            model: The specific model to use (optional). If not provided, the provider's default will be used.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            A Response object containing the response.
        """
        provider_instance = self._get_provider(provider)
        if model:
            kwargs['model'] = model
        return provider_instance.chat(messages, **kwargs)

    def stream_chat(
        self,
        provider: Union[str, Provider],
        messages: List[Union[Message, Dict[str, str]]],
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[Response]:
        """Stream a response for a conversation.

        Args:
            provider: The provider to use.
            messages: A list of messages in the conversation.
            model: The specific model to use (optional). If not provided, the provider's default will be used.
            **kwargs: Additional arguments to pass to the provider.

        Returns:
            An iterator of Response objects containing chunks of the response.
        """
        provider_instance = self._get_provider(provider)
        if model:
            kwargs['model'] = model
        return provider_instance.stream_chat(messages, **kwargs)