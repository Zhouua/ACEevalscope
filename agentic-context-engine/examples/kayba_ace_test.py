#!/usr/bin/env python3
"""
🌊 The Kayba Test - Self-Learning Demo 🌊
=========================================

Demonstrates ACE's ability to self-reflect and learn strategies
without external feedback. Tests the famous "seahorse emoji problem"
that confuses most LLMs.

Named after Kayba AI (海馬 kaiba = seahorse in Japanese).
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel

from ace import Generator, Reflector, Curator, Playbook
from ace.llm_providers import LiteLLMClient
from ace.observability import configure_opik

# Suppress LiteLLM debug messages
import litellm
litellm.suppress_debug_info = True

console = Console()


def main():
    # Display header
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]🌊 The Kayba Test - ACE Self-Learning Demo 🌊[/bold cyan]")
    console.print("[dim]Using Claude Opus 4.1[/dim]")
    console.print("=" * 60 + "\n")

    # Configure Opik observability
    integration = configure_opik(
        project_name="kayba-test",
        tags=["demo", "seahorse", "self-learning"]
    )
    status = "✓ Enabled" if integration.is_available() else "✗ Disabled"
    console.print(f"[cyan]Opik Observability: {status}[/cyan]")
    if integration.is_available():
        console.print("[dim]View traces at: https://www.comet.com/[/dim]")
    console.print()

    # Setup - Claude Opus 4.1 for transparent reasoning
    client = LiteLLMClient(
        model="claude-opus-4-1-20250805",
        temperature=0.7,
        max_tokens=4000,
        timeout=60,
    )

    # Using v1 prompts (default)
    generator = Generator(client)
    reflector = Reflector(client)
    curator = Curator(client)
    playbook = Playbook()

    question = "Give me the seahorse emoji?"

    # First ask
    console.print("[yellow]━━━ Round 1: Initial Query ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print("[dim]Playbook: Empty (no prior knowledge)[/dim]\n")
    output1 = generator.generate(
        question=question,
        context="",
        playbook=playbook,
    )
    console.print(f"[bold]Final Answer:[/bold] {output1.final_answer}")
    console.print(f"[dim]Reasoning: {output1.reasoning}[/dim]")

    # ACE self-reflects without external feedback
    console.print("\n[cyan]━━━ Self-Reflection Phase ━━━[/cyan]")
    console.print("[dim]ACE analyzes its own response without external feedback...[/dim]")
    console.print("[dim]Note: ACE has no ground truth - it doesn't know if the answer is correct![/dim]")

    # Reflect without feedback - just based on the output
    reflection = reflector.reflect(
        question=question,
        generator_output=output1,
        playbook=playbook,
        ground_truth=None,  # No ground truth provided
        feedback=None,  # No feedback provided
    )

    # Update playbook based on reflection
    curator_output = curator.curate(
        reflection=reflection,
        playbook=playbook,
        question_context="emoji questions",
        progress="self-learning",
    )
    playbook.apply_delta(curator_output.delta)

    if len(playbook.bullets()) > 0:
        console.print(
            f"[green]Playbook updated with {len(playbook.bullets())} self-learned insights[/green]"
        )
        console.print("\n[cyan]📚 Current Playbook:[/cyan]")
        console.print(Panel(playbook.as_prompt(), style="cyan"))

    # Second ask - should use playbook
    console.print(f"\n[yellow]━━━ Round 2: With Learned Strategies ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[dim]Playbook: {len(playbook.bullets())} learned strategies[/dim]\n")
    output2 = generator.generate(
        question=question,
        context="",
        playbook=playbook,
    )
    console.print(f"[bold]Final Answer:[/bold] {output2.final_answer}")
    console.print(f"[dim]Reasoning: {output2.reasoning}[/dim]")
    if output2.bullet_ids:
        console.print(f"[dim]Used bullets: {output2.bullet_ids}[/dim]")

    # Just show the two answers for comparison
    import time
    time.sleep(2)  # Pause before showing results
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📊 Results Comparison[/bold cyan]")
    console.print("=" * 60)

    console.print("\n[yellow]Round 1 Answer:[/yellow]")
    console.print(Panel(output1.final_answer, style="yellow"))

    console.print("[green]Round 2 Answer (with playbook):[/green]")
    console.print(Panel(output2.final_answer, style="green"))

    console.print("\n[bold red]⚠️  Fact Check:[/bold red]")
    console.print("[dim]There is NO seahorse emoji in Unicode (despite what models often claim).[/dim]")
    console.print("[dim]This demo shows how ACE learns strategies through self-reflection.[/dim]\n")


if __name__ == "__main__":
    main()
