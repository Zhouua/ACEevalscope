"""Example demonstrating playbook save and load functionality."""

from ace import Playbook, Sample, OfflineAdapter, Generator, Reflector, Curator
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient
import os


class SimpleTaskEnvironment(TaskEnvironment):
    """A simple environment for demonstration."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        # Simple evaluation: check if answer contains expected keyword
        is_correct = sample.ground_truth.lower() in generator_output.final_answer.lower()

        feedback = "Correct!" if is_correct else f"Expected '{sample.ground_truth}'"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth
        )


def train_and_save_playbook():
    """Train a playbook and save it to file."""
    print("Training playbook...")

    # Initialize components
    client = LiteLLMClient(model="gpt-3.5-turbo")
    generator = Generator(client)
    reflector = Reflector(client)
    curator = Curator(client)

    # Create offline adapter
    adapter = OfflineAdapter(
        generator=generator,
        reflector=reflector,
        curator=curator
    )

    # Create training samples
    samples = [
        Sample(
            question="What is the capital of France?",
            context="European geography",
            ground_truth="Paris"
        ),
        Sample(
            question="What is 2 + 2?",
            context="Basic arithmetic",
            ground_truth="4"
        ),
        Sample(
            question="What color is the sky on a clear day?",
            context="Common knowledge",
            ground_truth="blue"
        ),
    ]

    # Train with 2 epochs
    environment = SimpleTaskEnvironment()
    results = adapter.run(samples, environment, epochs=2)

    # Save the trained playbook
    playbook_path = "trained_playbook.json"
    adapter.playbook.save_to_file(playbook_path)

    print(f"✓ Trained playbook saved to {playbook_path}")
    print(f"  - Contains {len(adapter.playbook.bullets())} bullets")
    print(f"  - Sections: {list(adapter.playbook._sections.keys())}")

    return playbook_path


def load_and_use_playbook(playbook_path):
    """Load a pre-trained playbook and use it."""
    print(f"\nLoading playbook from {playbook_path}...")

    # Load the saved playbook
    playbook = Playbook.load_from_file(playbook_path)

    print(f"✓ Loaded playbook with {len(playbook.bullets())} bullets")

    # Use the loaded playbook with a new adapter
    client = LiteLLMClient(model="gpt-3.5-turbo")
    generator = Generator(client)

    # Generate answer using loaded playbook
    test_output = generator.generate(
        question="What is the capital of Germany?",
        context="European geography",
        playbook=playbook,
        reflection=None
    )

    print(f"\nTest question: What is the capital of Germany?")
    print(f"Generated answer: {test_output.final_answer}")
    print(f"Used bullets: {test_output.bullet_ids}")

    return playbook


def demonstrate_playbook_inspection(playbook):
    """Show how to inspect a loaded playbook."""
    print("\n" + "="*50)
    print("PLAYBOOK INSPECTION")
    print("="*50)

    # Print playbook statistics
    stats = playbook.stats()
    print(f"\nPlaybook Statistics:")
    print(f"  - Sections: {stats['sections']}")
    print(f"  - Total bullets: {stats['bullets']}")
    print(f"  - Helpful tags: {stats['tags']['helpful']}")
    print(f"  - Harmful tags: {stats['tags']['harmful']}")
    print(f"  - Neutral tags: {stats['tags']['neutral']}")

    # Show playbook as prompt (first 500 chars)
    prompt_view = playbook.as_prompt()
    if prompt_view:
        print(f"\nPlaybook as prompt (preview):")
        print("-" * 40)
        print(prompt_view[:500] + "..." if len(prompt_view) > 500 else prompt_view)

    # Show individual bullets
    print(f"\nIndividual bullets:")
    for bullet in playbook.bullets()[:3]:  # Show first 3 bullets
        print(f"  [{bullet.id}] {bullet.content[:60]}...")
        print(f"    Helpful: {bullet.helpful}, Harmful: {bullet.harmful}")


if __name__ == "__main__":
    try:
        # Step 1: Train and save a playbook
        playbook_path = train_and_save_playbook()

        # Step 2: Load and use the saved playbook
        loaded_playbook = load_and_use_playbook(playbook_path)

        # Step 3: Inspect the playbook
        demonstrate_playbook_inspection(loaded_playbook)

        print("\n✓ Example completed successfully!")

        # Clean up
        if os.path.exists(playbook_path):
            os.remove(playbook_path)
            print(f"✓ Cleaned up {playbook_path}")

    except ImportError:
        print("Note: This example requires an LLM provider to be configured.")
        print("Set your API key: export OPENAI_API_KEY='your-key'")
    except Exception as e:
        print(f"Example requires API keys to be set: {e}")

        # Demonstrate without API calls
        print("\n" + "="*50)
        print("DEMONSTRATING SAVE/LOAD WITHOUT API CALLS")
        print("="*50)

        # Create a playbook manually
        playbook = Playbook()
        playbook.add_bullet(
            section="general",
            content="Always provide step-by-step explanations",
            metadata={"helpful": 3, "harmful": 0}
        )
        playbook.add_bullet(
            section="math",
            content="Show your calculations clearly",
            metadata={"helpful": 5, "harmful": 0}
        )

        # Save it
        test_path = "test_playbook.json"
        playbook.save_to_file(test_path)
        print(f"\n✓ Saved playbook to {test_path}")

        # Load it back
        loaded = Playbook.load_from_file(test_path)
        print(f"✓ Loaded playbook with {len(loaded.bullets())} bullets")

        # Verify content matches
        for original, loaded_bullet in zip(playbook.bullets(), loaded.bullets()):
            assert original.content == loaded_bullet.content
            assert original.helpful == loaded_bullet.helpful
        print("✓ Content verified - save/load working correctly!")

        # Show the JSON structure
        print(f"\nJSON structure of saved playbook:")
        with open(test_path, "r") as f:
            print(f.read())

        # Clean up
        os.remove(test_path)
        print(f"\n✓ Cleaned up {test_path}")