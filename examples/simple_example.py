# Simple example to test the LLM Wrapper

from dotenv import load_dotenv
from llm_wrapper import LLMClient

# Load environment variables from .env file (if it exists)
load_dotenv()

# Initialize client from environment variables
# Note: You would need to set these environment variables or provide API keys directly
client = LLMClient.from_env()

def main():
    print("LLM Wrapper Test")
    print("----------------")
    
    # This will only work if you have the appropriate API keys set
    # Uncomment and use the provider you have an API key for
    
    # Example with OpenAI
    # response = client.complete(
    #     provider="openai",
    #     prompt="Tell me a joke about programming",
    #     max_tokens=100
    # )
    # print(response.text)
    
    print("\nIf you see this message without errors, the LLM Wrapper is installed correctly!")
    print("To use the wrapper with actual LLM providers, you'll need to set up API keys.")

if __name__ == "__main__":
    main()