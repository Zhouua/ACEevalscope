#!/usr/bin/env python3
"""
Example using ACE with LangChain integration.

This example demonstrates how to use the LangChainLiteLLMClient with ACE,
including router setup for load balancing across multiple models.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ace import (
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)

# Import LangChain client - will fail gracefully if not installed
try:
    from ace.llm_providers import LangChainLiteLLMClient
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain integration not available.")
    print("Install with: pip install ace-framework[langchain]")
    print("or: pip install langchain-litellm")

# Load environment variables
load_dotenv()


class SimpleQAEnvironment(TaskEnvironment):
    """Simple Q&A environment for testing."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = generator_output.final_answer

        if ground_truth.lower() in prediction.lower():
            feedback = "Correct!"
            score = 1.0
        else:
            feedback = f"Expected '{ground_truth}', but got '{prediction}'"
            score = 0.0

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"score": score}
        )


def example_basic_langchain():
    """Basic example using LangChain client."""
    print("\n=== Basic LangChain Example ===")

    if not LANGCHAIN_AVAILABLE:
        print("Skipping: LangChain not installed")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Create LangChain client
    client = LangChainLiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=256,
    )

    # Test direct completion
    response = client.complete("What is the capital of Japan?")
    print(f"Response: {response.text}")
    print(f"Model: {response.raw.get('model')}")


def example_with_router():
    """Example using router for load balancing."""
    print("\n=== LangChain Router Example ===")

    if not LANGCHAIN_AVAILABLE:
        print("Skipping: LangChain not installed")
        return

    try:
        from litellm import Router
    except ImportError:
        print("LiteLLM Router not available")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Define model list for router
    # In production, you might have multiple deployments of the same model
    model_list = [
        {
            "model_name": "primary",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        # You could add more endpoints here for load balancing
        # {
        #     "model_name": "primary",
        #     "litellm_params": {
        #         "model": "azure/gpt-35-turbo",
        #         "api_key": os.getenv("AZURE_API_KEY"),
        #         "api_base": os.getenv("AZURE_API_BASE"),
        #     },
        # },
    ]

    # Create router
    router = Router(model_list=model_list, routing_strategy="simple-shuffle")

    # Create client with router
    client = LangChainLiteLLMClient(
        model="primary",
        router=router,
        temperature=0.0,
    )

    # Test completion
    response = client.complete("Explain quantum computing in one sentence.")
    print(f"Response: {response.text}")
    if response.raw.get("router"):
        print(f"Used model: {response.raw.get('model_used')}")


def example_ace_with_langchain():
    """Example using ACE with LangChain client."""
    print("\n=== ACE with LangChain Example ===")

    if not LANGCHAIN_AVAILABLE:
        print("Skipping: LangChain not installed")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Create LangChain client
    client = LangChainLiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=512,
    )

    # Set up ACE components
    playbook = Playbook()
    generator = Generator(client)
    reflector = Reflector(client)
    curator = Curator(client)

    # Create adapter
    adapter = OfflineAdapter(
        playbook=playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=1,
    )

    # Create sample questions
    samples = [
        Sample(
            question="What is the largest planet in our solar system?",
            context="This is an astronomy question.",
            ground_truth="Jupiter",
        ),
        Sample(
            question="What is the speed of light in vacuum?",
            context="This is a physics question. Answer with the standard value.",
            ground_truth="299,792,458 meters per second",
        ),
        Sample(
            question="Who painted the Mona Lisa?",
            context="This is an art history question.",
            ground_truth="Leonardo da Vinci",
        ),
    ]

    # Run adaptation
    print("Running ACE adaptation with LangChain...")
    environment = SimpleQAEnvironment()
    results = adapter.run(samples, environment, epochs=1)

    print(f"\nProcessed {len(results)} samples")
    print(f"Final playbook has {len(playbook.bullets())} bullets")

    # Show some learned strategies
    if playbook.bullets():
        print("\nSample bullets from playbook:")
        for bullet in playbook.bullets()[:2]:
            print(f"- [{bullet.id}] {bullet.content[:100]}...")


async def example_async_streaming():
    """Example using async and streaming with LangChain."""
    print("\n=== Async Streaming Example ===")

    if not LANGCHAIN_AVAILABLE:
        print("Skipping: LangChain not installed")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Create LangChain client
    client = LangChainLiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
    )

    # Async completion
    print("Async completion:")
    response = await client.acomplete("Write a haiku about programming.")
    print(response.text)

    # Streaming
    print("\nStreaming response: ", end="")
    for token in client.complete_with_stream("Tell me a short joke about AI."):
        print(token, end="", flush=True)
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("ACE with LangChain Integration Examples")
    print("=" * 60)

    # Run examples
    example_basic_langchain()
    example_with_router()
    example_ace_with_langchain()

    # Run async example
    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        import asyncio
        asyncio.run(example_async_streaming())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()