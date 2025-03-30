"""Data models for the LLM Wrapper."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


@dataclass
class Message:
    """A message in a conversation with an LLM."""
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name")
        )


@dataclass
class Response:
    """A response from an LLM."""
    text: str
    provider: Provider
    model: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None

    @classmethod
    def from_provider_response(cls, provider: Provider, response: Dict[str, Any], model: str) -> "Response":
        """Create a Response from a provider-specific response."""
        if provider == Provider.OPENAI:
            return cls.from_openai(response, model)
        elif provider == Provider.GEMINI:
            return cls.from_gemini(response, model)
        elif provider == Provider.ANTHROPIC:
            return cls.from_anthropic(response, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def from_openai(cls, response: Dict[str, Any], model: str) -> "Response":
        """Create a Response from an OpenAI response."""
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = response.get("choices", [{}])[0].get("finish_reason")
        return cls(
            text=text,
            provider=Provider.OPENAI,
            model=model,
            raw_response=response,
            usage=response.get("usage"),
            finish_reason=finish_reason
        )

    @classmethod
    def from_gemini(cls, response: Dict[str, Any], model: str) -> "Response":
        """Create a Response from a Gemini response."""
        text = ""
        if "candidates" in response and response["candidates"]:
            content = response["candidates"][0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                text = parts[0].get("text", "")
        
        finish_reason = None
        if "candidates" in response and response["candidates"]:
            finish_reason = response["candidates"][0].get("finishReason")

        usage = None
        if "usageMetadata" in response:
            usage = {
                "prompt_tokens": response["usageMetadata"].get("promptTokenCount", 0),
                "completion_tokens": response["usageMetadata"].get("candidatesTokenCount", 0),
                "total_tokens": response["usageMetadata"].get("totalTokenCount", 0)
            }

        return cls(
            text=text,
            provider=Provider.GEMINI,
            model=model,
            raw_response=response,
            usage=usage,
            finish_reason=finish_reason
        )

    @classmethod
    def from_anthropic(cls, response: Dict[str, Any], model: str) -> "Response":
        """Create a Response from an Anthropic response."""
        text = response.get("content", [{}])[0].get("text", "")
        
        usage = None
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"].get("input_tokens", 0),
                "completion_tokens": response["usage"].get("output_tokens", 0),
                "total_tokens": response["usage"].get("input_tokens", 0) + 
                                response["usage"].get("output_tokens", 0)
            }

        return cls(
            text=text,
            provider=Provider.ANTHROPIC,
            model=model,
            raw_response=response,
            usage=usage,
            finish_reason=response.get("stop_reason")
        )