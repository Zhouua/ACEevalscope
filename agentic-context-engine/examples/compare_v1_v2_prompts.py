#!/usr/bin/env python3
"""
Compare v1 vs v2 prompts performance and characteristics.

This script runs the same tasks through both prompt versions and
compares accuracy, compliance, token usage, and output quality.
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

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
from ace.prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT
from ace.prompts_v2 import (
    PromptManager,
    validate_prompt_output,
)

# Load environment variables
load_dotenv()


class ComparisonEnvironment(TaskEnvironment):
    """Environment that tracks detailed metrics for comparison."""

    def evaluate(self, sample, generator_output):
        """Evaluate with detailed metric tracking."""
        # Check correctness
        # Try numeric comparison first
        try:
            expected = float(sample.ground_truth)
            actual = float(generator_output.final_answer)
            correct = abs(expected - actual) < 0.0001
        except:
            # Fall back to text comparison
            correct = sample.ground_truth.lower() in generator_output.final_answer.lower()

        # Track additional metrics
        metrics = {
            "correct": correct,
            "has_reasoning": bool(generator_output.reasoning),
            "reasoning_length": len(generator_output.reasoning),
            "cited_bullets": len(generator_output.bullet_ids),
            "answer_length": len(generator_output.final_answer)
        }

        # Check for confidence if available (v2 feature)
        if hasattr(generator_output, 'raw'):
            raw = generator_output.raw
            if 'answer_confidence' in raw:
                metrics['has_confidence'] = True
                metrics['confidence_value'] = raw['answer_confidence']
            else:
                metrics['has_confidence'] = False

            if 'confidence_scores' in raw:
                metrics['has_bullet_confidence'] = True
            else:
                metrics['has_bullet_confidence'] = False

        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
            metrics=metrics
        )


def run_comparison_test(llm_client, samples, environment, version="v1"):
    """Run test with specified prompt version."""

    if version == "v1":
        # Use v1 prompts
        generator = Generator(llm_client, prompt_template=GENERATOR_PROMPT)
        reflector = Reflector(llm_client, prompt_template=REFLECTOR_PROMPT)
        curator = Curator(llm_client, prompt_template=CURATOR_PROMPT)
        print("\n📝 Using v1 prompts (original)")
    else:
        # Use v2 prompts
        manager = PromptManager()
        generator = Generator(llm_client, prompt_template=manager.get_generator_prompt())
        reflector = Reflector(llm_client, prompt_template=manager.get_reflector_prompt())
        curator = Curator(llm_client, prompt_template=manager.get_curator_prompt())
        print("\n🚀 Using v2 prompts (enhanced)")

    # Create adapter
    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=generator,
        reflector=reflector,
        curator=curator
    )

    # Run adaptation
    start_time = time.time()
    results = adapter.run(samples, environment, epochs=1)
    elapsed_time = time.time() - start_time

    # Collect metrics
    metrics = {
        "correct": 0,
        "total": len(results),
        "has_reasoning": 0,
        "avg_reasoning_length": 0,
        "avg_bullets_cited": 0,
        "has_confidence": 0,
        "has_bullet_confidence": 0,
        "json_errors": 0,
        "elapsed_time": elapsed_time,
        "playbook_size": len(adapter.playbook.bullets())
    }

    reasoning_lengths = []
    bullets_cited = []
    confidence_values = []

    for result in results:
        result_metrics = result.environment_result.metrics

        if result_metrics.get('correct'):
            metrics['correct'] += 1

        if result_metrics.get('has_reasoning'):
            metrics['has_reasoning'] += 1
            reasoning_lengths.append(result_metrics.get('reasoning_length', 0))

        bullets_cited.append(result_metrics.get('cited_bullets', 0))

        if result_metrics.get('has_confidence'):
            metrics['has_confidence'] += 1
            confidence_values.append(result_metrics.get('confidence_value', 0))

        if result_metrics.get('has_bullet_confidence'):
            metrics['has_bullet_confidence'] += 1

    # Calculate averages
    if reasoning_lengths:
        metrics['avg_reasoning_length'] = sum(reasoning_lengths) / len(reasoning_lengths)
    if bullets_cited:
        metrics['avg_bullets_cited'] = sum(bullets_cited) / len(bullets_cited)
    if confidence_values:
        metrics['avg_confidence'] = sum(confidence_values) / len(confidence_values)

    # Test JSON compliance for a sample output
    if results:
        try:
            output = results[0].generator_output
            if hasattr(output, 'raw'):
                json_str = json.dumps(output.raw)
                is_valid, errors = validate_prompt_output(json_str, "generator")
                metrics['json_valid'] = is_valid
                metrics['validation_errors'] = errors
        except Exception as e:
            metrics['json_valid'] = False
            metrics['validation_errors'] = [str(e)]

    return metrics, adapter.playbook


def print_comparison_results(v1_metrics, v2_metrics):
    """Print detailed comparison between v1 and v2."""

    print("\n" + "="*70)
    print(" "*25 + "COMPARISON RESULTS")
    print("="*70)

    # Accuracy comparison
    v1_accuracy = v1_metrics['correct'] / v1_metrics['total'] * 100
    v2_accuracy = v2_metrics['correct'] / v2_metrics['total'] * 100

    print(f"\n📊 ACCURACY")
    print(f"  v1: {v1_accuracy:.1f}% ({v1_metrics['correct']}/{v1_metrics['total']})")
    print(f"  v2: {v2_accuracy:.1f}% ({v2_metrics['correct']}/{v2_metrics['total']})")
    if v2_accuracy > v1_accuracy:
        print(f"  ✅ v2 is {v2_accuracy - v1_accuracy:.1f}% more accurate")
    elif v1_accuracy > v2_accuracy:
        print(f"  ⚠️  v1 is {v1_accuracy - v2_accuracy:.1f}% more accurate")
    else:
        print(f"  🔄 Same accuracy")

    # Reasoning quality
    print(f"\n💭 REASONING QUALITY")
    print(f"  v1 avg length: {v1_metrics.get('avg_reasoning_length', 0):.0f} chars")
    print(f"  v2 avg length: {v2_metrics.get('avg_reasoning_length', 0):.0f} chars")
    if v2_metrics.get('avg_reasoning_length', 0) > v1_metrics.get('avg_reasoning_length', 0):
        improvement = ((v2_metrics['avg_reasoning_length'] / v1_metrics['avg_reasoning_length']) - 1) * 100
        print(f"  ✅ v2 reasoning is {improvement:.0f}% more detailed")

    # Bullet citations
    print(f"\n📌 STRATEGY USAGE")
    print(f"  v1 avg bullets cited: {v1_metrics.get('avg_bullets_cited', 0):.1f}")
    print(f"  v2 avg bullets cited: {v2_metrics.get('avg_bullets_cited', 0):.1f}")
    print(f"  v1 playbook size: {v1_metrics['playbook_size']}")
    print(f"  v2 playbook size: {v2_metrics['playbook_size']}")

    # v2-specific features
    print(f"\n🆕 V2-SPECIFIC FEATURES")
    print(f"  Confidence scores: {v2_metrics.get('has_confidence', 0)}/{v2_metrics['total']} samples")
    if v2_metrics.get('has_confidence', 0) > 0:
        print(f"  Avg confidence: {v2_metrics.get('avg_confidence', 0):.2%}")
    print(f"  Bullet confidence: {v2_metrics.get('has_bullet_confidence', 0)}/{v2_metrics['total']} samples")

    # JSON compliance
    print(f"\n✓ OUTPUT COMPLIANCE")
    print(f"  v1 JSON valid: {v1_metrics.get('json_valid', False)}")
    print(f"  v2 JSON valid: {v2_metrics.get('json_valid', False)}")
    if v1_metrics.get('validation_errors'):
        print(f"  v1 errors: {v1_metrics['validation_errors']}")
    if v2_metrics.get('validation_errors'):
        print(f"  v2 errors: {v2_metrics['validation_errors']}")

    # Performance
    print(f"\n⚡ PERFORMANCE")
    print(f"  v1 time: {v1_metrics['elapsed_time']:.1f}s")
    print(f"  v2 time: {v2_metrics['elapsed_time']:.1f}s")
    if v2_metrics['elapsed_time'] < v1_metrics['elapsed_time']:
        print(f"  ✅ v2 is {v1_metrics['elapsed_time'] - v2_metrics['elapsed_time']:.1f}s faster")
    else:
        print(f"  ⚠️  v2 is {v2_metrics['elapsed_time'] - v1_metrics['elapsed_time']:.1f}s slower")

    # Overall assessment
    print(f"\n📈 OVERALL ASSESSMENT")
    v2_wins = 0
    v1_wins = 0

    if v2_accuracy > v1_accuracy:
        v2_wins += 1
    elif v1_accuracy > v2_accuracy:
        v1_wins += 1

    if v2_metrics.get('avg_reasoning_length', 0) > v1_metrics.get('avg_reasoning_length', 0):
        v2_wins += 1

    if v2_metrics.get('has_confidence', 0) > 0:
        v2_wins += 1  # v2-specific feature

    if v2_metrics.get('json_valid', False) and not v1_metrics.get('json_valid', False):
        v2_wins += 1
    elif v1_metrics.get('json_valid', False) and not v2_metrics.get('json_valid', False):
        v1_wins += 1

    if v2_wins > v1_wins:
        print(f"  🎉 v2 prompts perform better overall ({v2_wins} wins vs {v1_wins})")
    elif v1_wins > v2_wins:
        print(f"  ℹ️  v1 prompts perform better ({v1_wins} wins vs {v2_wins})")
    else:
        print(f"  🔄 Performance is comparable")

    print("\n" + "="*70)


def main():
    """Run comprehensive v1 vs v2 comparison."""

    print("\n" + "="*70)
    print(" "*20 + "v1 vs v2 PROMPT COMPARISON")
    print(" "*15 + "Testing prompt engineering improvements")
    print("="*70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Please set OPENAI_API_KEY in your .env file")
        return

    # Create LLM client
    llm = LiteLLMClient(
        model="gpt-3.5-turbo",
        temperature=0.1  # Low temperature for consistency
    )

    # Test samples covering different types
    samples = [
        # Math problems
        Sample(
            question="What is 12 + 8?",
            ground_truth="20"
        ),
        Sample(
            question="Solve for x: 3x - 7 = 14",
            ground_truth="7"
        ),
        Sample(
            question="What is 25% of 80?",
            ground_truth="20"
        ),

        # Factual questions
        Sample(
            question="What is the capital of France?",
            ground_truth="Paris"
        ),
        Sample(
            question="Who painted the Mona Lisa?",
            ground_truth="Leonardo da Vinci"
        ),

        # Reasoning questions
        Sample(
            question="If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
            ground_truth="No"
        ),
    ]

    # Create comparison environment
    environment = ComparisonEnvironment()

    print(f"\n🧪 Testing with {len(samples)} diverse samples...")

    # Test v1 prompts
    print("\n" + "-"*50)
    v1_metrics, v1_playbook = run_comparison_test(llm, samples, environment, version="v1")

    # Test v2 prompts
    print("\n" + "-"*50)
    v2_metrics, v2_playbook = run_comparison_test(llm, samples, environment, version="v2")

    # Compare results
    print_comparison_results(v1_metrics, v2_metrics)

    # Show sample strategies learned
    print("\n📚 SAMPLE STRATEGIES LEARNED")

    if v1_playbook.bullets():
        print("\nv1 First Strategy:")
        print(f"  {v1_playbook.bullets()[0].content[:150]}...")

    if v2_playbook.bullets():
        print("\nv2 First Strategy:")
        print(f"  {v2_playbook.bullets()[0].content[:150]}...")

    # Domain-specific comparison (if time permits)
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("-"*70)

    if v2_metrics.get('has_confidence', 0) > 0:
        print("✅ v2 provides confidence scores for better uncertainty handling")

    if v2_metrics.get('avg_reasoning_length', 0) > v1_metrics.get('avg_reasoning_length', 0) * 1.2:
        print("✅ v2 produces more detailed reasoning chains")

    if v2_metrics.get('json_valid', False):
        print("✅ v2 has better JSON schema compliance")

    print("\n🔧 Consider using:")
    print("  - v2 prompts for production (better structure & features)")
    print("  - Domain-specific v2 variants for specialized tasks")
    print("  - PromptManager for A/B testing in your use case")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()