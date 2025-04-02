# Simple example to test the LLM Wrapper

from dotenv import load_dotenv
from llm_wrapper import LLMClient
import os

# Load environment variables from .env file (if it exists)
load_dotenv()

def main():
    print("LLM Wrapper Test")
    print("----------------")
    
    # This will use dummy API keys for demonstration purposes
    # In real usage, you should use your actual API keys
    try:
        # First try environment variables
        client = LLMClient.from_env()
        print("Successfully initialized client from environment variables")
    except ValueError:
        # Fall back to dummy keys for demonstration
        print("No API keys found in environment. Using dummy keys for demonstration.")
        
        # NOTE: These are fake API keys for demonstration only!
        # The client will initialize but API calls will fail
        client = LLMClient(
            openai_api_key="dummy_openai_key_for_demonstration",
            gemini_api_key="AIzaSyChxCD66cgwmyej5BjlW3I-dluOL2ckD_o",
            anthropic_api_key="dummy_anthropic_key_for_demonstration"
        )
    
    print("\nAvailable providers:")
    for provider in client.providers:
        print(f"- {provider}")
    
    print("\nIf you see this message without errors, the LLM Wrapper is installed correctly!")
    print("To use the wrapper with actual LLM providers, you'll need to set up valid API keys.")
    print("Example usage in your own code:")
    print("Testing Gemini API key with a real request:")
    # Test with Gemini API
    response = client.complete(
        provider="gemini",
        prompt="Tell me a joke about programming"
    )
    print(response.text)

if __name__ == "__main__":
    main()