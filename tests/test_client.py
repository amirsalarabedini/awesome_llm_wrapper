"""Tests for the LLM Wrapper client."""

import os
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_wrapper import LLMClient
from src.llm_wrapper.models import Provider


class TestLLMClient(unittest.TestCase):
    """Test cases for the LLMClient class."""

    def test_init_with_keys(self):
        """Test initializing the client with API keys."""
        client = LLMClient(
            openai_api_key="test-openai-key",
            gemini_api_key="test-gemini-key",
            anthropic_api_key="test-anthropic-key"
        )
        
        self.assertIn(Provider.OPENAI, client.providers)
        self.assertIn(Provider.GEMINI, client.providers)
        self.assertIn(Provider.ANTHROPIC, client.providers)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "env-openai-key",
        "GEMINI_API_KEY": "env-gemini-key",
        "ANTHROPIC_API_KEY": "env-anthropic-key"
    })
    def test_init_from_env(self):
        """Test initializing the client from environment variables."""
        client = LLMClient.from_env()
        
        self.assertIn(Provider.OPENAI, client.providers)
        self.assertIn(Provider.GEMINI, client.providers)
        self.assertIn(Provider.ANTHROPIC, client.providers)