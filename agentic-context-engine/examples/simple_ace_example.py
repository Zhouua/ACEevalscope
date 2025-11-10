#!/usr/bin/env python3
"""
Simplest possible ACE example with LiteLLM.

This shows the minimal code needed to use ACE with a production LLM.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
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
)

# Load environment variables
load_dotenv()


class SimpleEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # 1. Create LLM client
    llm = LiteLLMClient(model="gpt-3.5-turbo")

    # 2. Create ACE components
    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
    )

    # 3. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 4. Run adaptation
    environment = SimpleEnvironment()
    results = adapter.run(samples, environment, epochs=1)

    # 5. Check results
    print(f"Trained on {len(results)} samples")
    print(f"Playbook now has {len(adapter.playbook.bullets())} strategies")

    # Show a few learned strategies
    for bullet in adapter.playbook.bullets()[:2]:
        print(f"\nLearned: {bullet.content}")


if __name__ == "__main__":
    main()