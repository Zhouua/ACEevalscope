#!/usr/bin/env python3
"""
Quick start example using LiteLLM with various providers.

This example demonstrates how to use ACE with different LLM providers
through the unified LiteLLM interface.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import ace
sys.path.append(str(Path(__file__).parent.parent))

from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
    LITELLM_AVAILABLE,
)


# Load environment variables from .env file
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
            feedback = f"Expected something about '{ground_truth}', but got '{prediction}'"
            score = 0.0

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"score": score}
        )


def example_openai():
    """Example using OpenAI GPT models."""
    print("\n=== OpenAI GPT Example ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Create client for OpenAI
    client = LiteLLMClient(
        model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
        temperature=0.0,
        max_tokens=256,
    )

    # Test direct completion
    response = client.complete("What is the capital of France?")
    print(f"Response: {response.text}")
    print(f"Provider: {response.raw.get('provider')}")
    print(f"Model: {response.raw.get('model')}")


def example_anthropic():
    """Example using Anthropic Claude models."""
    print("\n=== Anthropic Claude Example ===")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Create client for Claude
    client = LiteLLMClient(
        model="claude-3-haiku-20240307",  # or "claude-3-sonnet-20240229"
        temperature=0.0,
        max_tokens=256,
    )

    # Test direct completion
    response = client.complete("Explain quantum computing in one sentence.")
    print(f"Response: {response.text}")
    print(f"Provider: {response.raw.get('provider')}")


def example_with_fallbacks():
    """Example using fallback models for resilience."""
    print("\n=== Fallback Example ===")

    # Create client with fallbacks
    # If primary model fails, it will try fallback models
    client = LiteLLMClient(
        model="gpt-4",  # Primary (more expensive)
        fallbacks=[
            "gpt-3.5-turbo",  # First fallback (cheaper)
            "claude-3-haiku-20240307",  # Second fallback
        ],
        temperature=0.0,
        max_tokens=256,
    )

    response = client.complete("What is 2 + 2?")
    print(f"Response: {response.text}")
    print(f"Used model: {response.raw.get('model')}")


def example_ace_adaptation():
    """Example using ACE adaptation with LiteLLM."""
    print("\n=== ACE Adaptation Example ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Create LiteLLM client
    client = LiteLLMClient(
        model="gpt-4o-mini",
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
            question="What is the capital of France?",
            context="This is a geography question.",
            ground_truth="Paris",
        ),
        Sample(
            question="What is 2 + 2?",
            context="This is a math question.",
            ground_truth="4",
        ),
        Sample(
            question="What color is the sky on a clear day?",
            context="This is about weather and atmosphere.",
            ground_truth="blue",
        ),
    ]

    # Run adaptation
    print("Running adaptation...")
    environment = SimpleQAEnvironment()
    results = adapter.run(samples, environment, epochs=1)

    print(f"\nProcessed {len(results)} samples")
    print(f"Final playbook has {len(playbook.bullets())} bullets")

    # Show some bullets
    print("\nSample bullets from playbook:")
    for bullet in playbook.bullets()[:3]:
        print(f"- [{bullet.id}] {bullet.content[:100]}...")


def example_streaming():
    """Example with streaming responses."""
    print("\n=== Streaming Example ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    client = LiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
    )

    print("Streaming response: ", end="")
    for chunk in client.complete_with_stream("Tell me a short joke about programming"):
        print(chunk, end="", flush=True)
    print()  # New line at the end


async def example_async():
    """Example with async completion."""
    print("\n=== Async Example ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    client = LiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=50,
    )

    # Async completion
    response = await client.acomplete("What is the meaning of life?")
    print(f"Async Response: {response.text}")


def main():
    """Run all examples."""

    if not LITELLM_AVAILABLE:
        print("LiteLLM is not installed. Please run: pip install litellm")
        return

    print("=" * 60)
    print("ACE with LiteLLM Examples")
    print("=" * 60)

    # Check for at least one API key
    has_any_key = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("COHERE_API_KEY"),
    ])

    if not has_any_key:
        print("\nNo API keys found!")
        print("Please copy .env.example to .env and add your API keys.")
        print("\nAt minimum, you need one of:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        print("  - COHERE_API_KEY")
        return

    # Run examples based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        example_openai()
        example_streaming()
        example_ace_adaptation()

    if os.getenv("ANTHROPIC_API_KEY"):
        example_anthropic()

    # Run fallback example if we have multiple providers
    if os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
        example_with_fallbacks()

    # Run async example
    if os.getenv("OPENAI_API_KEY"):
        import asyncio
        asyncio.run(example_async())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()