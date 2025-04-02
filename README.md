# LLM Wrapper

A unified Python library for interacting with multiple Large Language Model APIs including OpenAI, Google Gemini, and Anthropic. This wrapper provides a consistent interface across different LLM providers, simplifying integration and allowing easy switching between models.

## Features

- Single interface for multiple LLM providers (OpenAI, Google Gemini, Anthropic)
- Simple configuration with API keys or environment variables
- Standardized request and response formats across providers
- Comprehensive error handling with detailed contextual messages
- Automatic rate limit handling with exponential backoff
- Full streaming support for real-time responses
- Proper system message handling for all providers
- Input validation and type checking for robust operation
- Detailed usage metrics for token consumption
- Consistent default parameters across all providers
- Consistent parameter handling across providers
- Built-in error handling with retries for rate limiting
- Streaming support for efficient token generation
- Simplified message handling for both completions and chat

## Installation

```bash
pip install awesome-llm-wrapper
```

## Quick Start

```python
from llm_wrapper import LLMClient

# Initialize with your API keys
client = LLMClient(
    openai_api_key="your-openai-key",
    gemini_api_key="your-gemini-key",
    anthropic_api_key="your-anthropic-key"
)

# Or load from environment variables
client = LLMClient.from_env()

# Make a simple completion request
response = client.complete(
    provider="openai",  # or "gemini" or "anthropic"
    prompt="Tell me a joke about programming",
    model="gpt-4o-mini",  # optional: specify a model (defaults to provider's DEFAULT_MODEL if not specified)
    max_tokens=100
)

# Access the generated text
print(response.text)

# Access additional information
print(f"Model used: {response.model}")
print(f"Provider: {response.provider}")
print(f"Token usage: {response.usage}")
```

## Advanced Usage

### Chat Completions

```python
# Chat completion with conversation history
response = client.chat(
    provider="anthropic",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is its population?"}
    ],
    model="claude-3-5-sonnet",  # optional: specify a model (defaults to provider's DEFAULT_MODEL if not specified)
    temperature=0.7,
    max_tokens=150
)

print(response.text)
```

### Streaming Responses

```python
# Streaming response for real-time output
for chunk in client.stream_chat(
    provider="openai",
    messages=[
        {"role": "user", "content": "Write a poem about AI"}
    ],
    model="gpt-4o-mini",  # optional: specify a model (defaults to provider's DEFAULT_MODEL if not specified)
    temperature=0.9
):
    print(chunk.text, end="")
```

### Using Message Objects

```python
from llm_wrapper import LLMClient, Message

# Using Message objects instead of dictionaries
response = client.chat(
    provider="gemini",
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain quantum computing in simple terms.")
    ],
    model="gemini-2.0-flash"  # optional: specify a model
)
```

## System Message Handling

Each provider handles system messages differently, but LLM Wrapper normalizes this for you:

```python
# System messages are properly formatted for each provider
response = client.chat(
    provider="anthropic",  # Works the same for OpenAI and Gemini
    messages=[
        {"role": "system", "content": "You are a helpful assistant who always responds in rhyme."},
        {"role": "user", "content": "Tell me about machine learning."}
    ]
)
```

Behind the scenes:
- **OpenAI**: System messages are passed directly as "system" role messages
- **Anthropic**: System messages are formatted with `<system>...</system>` tags
- **Gemini**: System messages are prepended to the first user message

## Model Selection

Each provider has a default model, but you can specify a different model using the `model` parameter:

```python
# Using specific models for each provider

# OpenAI (default: gpt-4o-mini)
response = client.complete(
    provider="openai",
    prompt="Generate a story",
    model="gpt-4",  # override the default model
    max_tokens=100
)

# Gemini (default: gemini-2.0-flash)
response = client.complete(
    provider="gemini",
    prompt="Generate a story",
    model="gemini-2.0-pro",
    max_tokens=100
)

# Anthropic (default: claude-3-5-sonnet)
response = client.complete(
    provider="anthropic",
    prompt="Generate a story",
    model="claude-3-opus",
    max_tokens=100
)
```

## Error Handling and Rate Limiting

The library provides comprehensive error handling, including automatic retries for rate limit errors:

```python
from llm_wrapper import LLMClient

try:
    response = client.complete(
        provider="openai",
        prompt="Generate a story",
        max_tokens=100
    )
    print(response.text)
except ValueError as e:
    print(f"Configuration error: {e}")  # Invalid API keys, missing parameters, etc.
except requests.HTTPError as e:
    print(f"HTTP error: {e}")  # API errors with detailed context
except Exception as e:
    print(f"Other error: {e}")  # Any other errors that might occur
```

Rate limit handling is built-in:
- Automatic retries with exponential backoff
- Respects the `Retry-After` header when provided by the API
- Detailed error messages if all retries fail

## Configuration

### Environment Variables

You can configure the library using environment variables:

```
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
ANTHROPIC_API_KEY=your-anthropic-key
```

If no valid API keys are found, the library will raise a clear error message.

### Available Models

Each provider has a default model that will be used if no model is specified:

| Provider  | Default Model       | Other Available Models                      |
|-----------|---------------------|--------------------------------------------|
| OpenAI    | gpt-4o-mini         | gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.    |
| Gemini    | gemini-2.0-flash    | gemini-2.0-pro, gemini-1.5-pro, etc.       |
| Anthropic | claude-3-5-sonnet   | claude-3-opus, claude-3-haiku, etc.        |

You can specify any model supported by the provider's API using the `model` parameter.

### Consistent Default Parameters

All providers use consistent default parameters:
- `max_tokens`: 1000 (for all providers)
- Other parameters are passed directly to the underlying API

### Response Object

The `Response` object provides access to:

- `text`: The generated text content
- `provider`: The provider used (OpenAI, Gemini, or Anthropic)
- `model`: The specific model used
- `raw_response`: The full raw response from the provider
- `usage`: Token usage statistics
- `finish_reason`: The reason why the generation stopped

## Parameter Handling

The library intelligently handles parameter conversion between different providers:

### Common Parameters (automatically converted)

| Parameter | OpenAI | Gemini | Anthropic |
|-----------|--------|--------|-----------|
| `max_tokens` | `max_tokens` | `maxOutputTokens` | `max_tokens` |
| `temperature` | `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `topP` | `top_p` |
| `top_k` | N/A | `topK` | N/A |

You can use the snake_case (Python style) parameter names consistently across all providers, and the library will automatically convert them to the appropriate format for each API:

```python
# This works for all providers
response = client.complete(
    provider="gemini",  # or "openai" or "anthropic" 
    prompt="Tell me a joke",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=40  # only used by Gemini, ignored by others
)
```

### Provider-Specific Parameters

You can also pass provider-specific parameters directly:

```python
# OpenAI-specific
response = client.complete(
    provider="openai",
    prompt="Tell me a joke",
    frequency_penalty=0.5,  # OpenAI-specific
    presence_penalty=0.2    # OpenAI-specific
)

# Gemini-specific
response = client.complete(
    provider="gemini",
    prompt="Tell me a joke",
    candidateCount=2,  # Gemini-specific
    stopSequences=["The end"]  # Gemini-specific
)

# Anthropic-specific
response = client.complete(
    provider="anthropic",
    prompt="Tell me a joke",
    stop_sequences=["The end"]  # Anthropic-specific
)
```

## Development

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-wrapper.git
   cd llm-wrapper
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

### Running Tests

Run the test suite with:
```bash
pytest tests/
```

### Project Structure

- `src/llm_wrapper/`: Main package directory
  - `client.py`: Main client interface
  - `models.py`: Data models and response objects
  - `providers/`: Provider implementations
    - `base.py`: Base provider class
    - `openai.py`: OpenAI provider
    - `gemini.py`: Google Gemini provider
    - `anthropic.py`: Anthropic provider

### Adding New Providers

1. Create a new provider file in `src/llm_wrapper/providers/` (e.g., `new_provider.py`)
2. Implement the provider class inheriting from `BaseProvider`
3. Add the provider to the `Provider` enum in `models.py`
4. Update the `LLMClient` class in `client.py` to support the new provider
5. Add tests for the new provider in `tests/`

### Contributing Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive docstrings for all public methods
- Include unit tests for new features
- Update documentation (README.md) when adding new features
- Use descriptive commit messages

## License

MIT