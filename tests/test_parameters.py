"""Tests for parameter handling in LLM Wrapper providers."""

import unittest
from unittest.mock import patch, MagicMock
import json

from llm_wrapper.providers.openai import OpenAIProvider
from llm_wrapper.providers.gemini import GeminiProvider
from llm_wrapper.providers.anthropic import AnthropicProvider


class ParameterHandlingTest(unittest.TestCase):
    """Test parameter handling for all providers."""
    
    def setUp(self):
        """Set up the test."""
        self.openai = OpenAIProvider("dummy_key")
        self.gemini = GeminiProvider("dummy_key")
        self.anthropic = AnthropicProvider("dummy_key")
        
    @patch("requests.post")
    def test_openai_parameters(self, mock_post):
        """Test that OpenAI parameters are correctly formatted."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        # Call the method with parameters
        self.openai.chat([{"role": "user", "content": "Test"}], max_tokens=100, temperature=0.7)
        
        # Get the payload that was sent
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        
        # Assert that parameters were formatted correctly
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.7)
        
    @patch("requests.post")
    def test_gemini_parameters(self, mock_post):
        """Test that Gemini parameters are correctly formatted."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Test response"}]}}]
        }
        mock_post.return_value = mock_response
        
        # Call the method with parameters
        self.gemini.chat([{"role": "user", "content": "Test"}], max_tokens=100, temperature=0.7, top_p=0.9, top_k=40)
        
        # Get the payload that was sent
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        
        # Assert that parameters were formatted correctly
        self.assertEqual(payload["maxOutputTokens"], 100)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["topP"], 0.9)
        self.assertEqual(payload["topK"], 40)
        
    @patch("requests.post")
    def test_anthropic_parameters(self, mock_post):
        """Test that Anthropic parameters are correctly formatted."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Test response"}]
        }
        mock_post.return_value = mock_response
        
        # Call the method with parameters
        self.anthropic.chat([{"role": "user", "content": "Test"}], max_tokens=100, temperature=0.7)
        
        # Get the payload that was sent
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        
        # Assert that parameters were formatted correctly
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main() 