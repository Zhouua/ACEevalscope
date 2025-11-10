# 📚 ACE Framework API Reference

Complete API documentation for the ACE Framework.

## Core Components

### Generator

The Generator produces answers using the current playbook of strategies.

```python
from ace import Generator, LiteLLMClient

client = LiteLLMClient(model="gpt-4")
generator = Generator(client)

output = generator.generate(
    question="What is 2+2?",
    context="Show your work",
    playbook=playbook,
    reflection=None  # Optional reflection from previous attempt
)

# Output contains:
# - output.final_answer: The generated answer
# - output.reasoning: Step-by-step reasoning
# - output.bullet_ids: List of playbook strategies used
```

### Reflector

The Reflector analyzes what went right or wrong and tags which strategies helped or hurt.

```python
from ace import Reflector

reflector = Reflector(client)

reflection = reflector.reflect(
    question="What is 2+2?",
    generator_output=output,
    playbook=playbook,
    ground_truth="4",
    feedback="Correct!",
    max_refinement_rounds=1
)

# Reflection contains:
# - reflection.reasoning: Analysis of the outcome
# - reflection.error_identification: What went wrong (if anything)
# - reflection.root_cause_analysis: Why it went wrong
# - reflection.correct_approach: What should have been done
# - reflection.key_insight: Main lesson learned
# - reflection.bullet_tags: List of (bullet_id, tag) pairs
```

### Curator

The Curator transforms reflections into playbook updates.

```python
from ace import Curator

curator = Curator(client)

curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="Math problems",
    progress="3/5 correct"
)

# Apply the updates
playbook.apply_delta(curator_output.delta)
```

## Playbook Management

### Creating a Playbook

```python
from ace import Playbook

playbook = Playbook()

# Add a strategy
bullet = playbook.add_bullet(
    section="Math Strategies",
    content="Break complex problems into smaller steps",
    metadata={"helpful": 5, "harmful": 0, "neutral": 1}
)
```

### Saving and Loading

```python
# Save to file
playbook.save_to_file("my_strategies.json")

# Load from file
loaded_playbook = Playbook.load_from_file("my_strategies.json")
```

### Playbook Statistics

```python
stats = playbook.stats()
# Returns:
# {
#   "sections": 3,
#   "bullets": 15,
#   "tags": {
#     "helpful": 45,
#     "harmful": 5,
#     "neutral": 10
#   }
# }
```

## Adapters

### OfflineAdapter

Train on a batch of samples.

```python
from ace import OfflineAdapter
from ace.types import Sample

adapter = OfflineAdapter(generator, reflector, curator)

samples = [
    Sample(
        question="What is 2+2?",
        context="Calculate",
        ground_truth="4"
    ),
    # More samples...
]

results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,
    verbose=True
)
```

### OnlineAdapter

Learn from tasks one at a time.

```python
from ace import OnlineAdapter

adapter = OnlineAdapter(
    playbook=existing_playbook,
    generator=generator,
    reflector=reflector,
    curator=curator
)

for task in tasks:
    result = adapter.process(task, environment)
    # Playbook updates automatically after each task
```

## Environments

### Creating Environments

All environments should extend the `TaskEnvironment` base class.

#### Simple Environment Example

Basic environment that compares output to ground truth using substring matching:

```python
from ace import TaskEnvironment, EnvironmentResult

class SimpleEnvironment(TaskEnvironment):
    """Basic environment for testing - checks if ground truth appears in answer."""

    def evaluate(self, sample, generator_output):
        # Simple substring matching (case-insensitive)
        correct = str(sample.ground_truth).lower() in str(generator_output.final_answer).lower()

        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )

# Usage
env = SimpleEnvironment()
result = env.evaluate(sample, generator_output)
```

### Custom Environments

```python
from ace import TaskEnvironment, EnvironmentResult

class CodeEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        # Run the code
        success = execute_code(output.final_answer)

        return EnvironmentResult(
            feedback="Tests passed" if success else "Tests failed",
            ground_truth=sample.ground_truth,
            metrics={"pass_rate": 1.0 if success else 0.0}
        )
```

## LLM Clients

### LiteLLMClient

Support for 100+ LLM providers.

```python
from ace import LiteLLMClient

# Basic usage
client = LiteLLMClient(model="gpt-4")

# With configuration
client = LiteLLMClient(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    fallbacks=["claude-3-haiku", "gpt-3.5-turbo"]
)

# Generate completion
response = client.complete("What is the meaning of life?")
print(response.text)
```

### LangChainLiteLLMClient

Integration with LangChain.

```python
from ace.llm_providers import LangChainLiteLLMClient

client = LangChainLiteLLMClient(
    model="gpt-4",
    tags=["production"],
    metadata={"user": "alice"}
)
```

## Types

### Sample

```python
from ace.types import Sample

sample = Sample(
    question="Your question here",
    context="Optional context or requirements",
    ground_truth="Expected answer (optional)"
)
```

### GeneratorOutput

```python
@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]
```

### ReflectorOutput

```python
@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]
```

### EnvironmentResult

```python
@dataclass
class EnvironmentResult:
    feedback: str
    ground_truth: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
```

## Delta Operations

### DeltaOperation Types

- `ADD`: Add new bullet to playbook
- `UPDATE`: Update existing bullet content
- `TAG`: Update helpful/harmful/neutral counts
- `REMOVE`: Remove bullet from playbook

```python
from ace.delta import DeltaOperation

op = DeltaOperation(
    type="ADD",
    section="Math Strategies",
    content="Always check your work",
    bullet_id="math-00001"
)
```

## Prompts

### Using Default Prompts

```python
from ace.prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT

generator = Generator(client, prompt_template=GENERATOR_PROMPT)
```

### Using v2 Prompts (Experimental)

```python
from ace.prompts_v2 import PromptManager

manager = PromptManager(default_version="2.0")

generator = Generator(
    client,
    prompt_template=manager.get_generator_prompt(domain="math")
)
```

### Custom Prompts

```python
custom_prompt = '''
Playbook: {playbook}
Question: {question}
Context: {context}

Generate a JSON response with:
- reasoning: Your step-by-step thought process
- bullet_ids: List of playbook IDs you used
- final_answer: Your answer
'''

generator = Generator(client, prompt_template=custom_prompt)
```

## Async Operations

```python
import asyncio

async def main():
    # Async completion
    response = await client.acomplete("What is 2+2?")

    # Async adapter operations also supported
    # (Implementation depends on adapter async support)

asyncio.run(main())
```

## Streaming

```python
# Stream responses token by token
for chunk in client.complete_with_stream("Write a story"):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from ace.exceptions import ACEException

try:
    output = generator.generate(...)
except ACEException as e:
    print(f"ACE error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GOOGLE_API_KEY="your-key"

# Custom endpoint
export LITELLM_API_BASE="https://your-endpoint.com"
```

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or just for ACE
logging.getLogger("ace").setLevel(logging.DEBUG)
```

## Best Practices

1. **Start with SimpleEnvironment**: Get basic training working first
2. **Use fallback models**: Ensure reliability in production
3. **Save playbooks regularly**: Preserve learned strategies
4. **Monitor costs**: Track token usage with metrics
5. **Test with dummy mode**: Validate logic without API calls
6. **Use appropriate epochs**: 2-3 epochs usually sufficient
7. **Implement custom environments**: Tailor evaluation to your task

## Examples

See the [examples](../examples/) directory for complete working examples:

- `simple_ace_example.py` - Basic usage
- `quickstart_litellm.py` - LiteLLM setup
- `langchain_example.py` - LangChain integration
- `playbook_persistence.py` - Save/load strategies
- `advanced_prompts_v2.py` - v2 prompt features
- `compare_v1_v2_prompts.py` - Performance comparison