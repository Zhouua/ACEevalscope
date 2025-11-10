## Quick Start

### Installation
```bash
# Basic installation
pip install ace-framework

# With optional features
pip install ace-framework[demos]         # Browser automation demos
pip install ace-framework[transformers]  # Local model support
pip install ace-framework[all]          # All features

# Contributors (10-100x faster with UV)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync
```

### Set API Key
```bash
export OPENAI_API_KEY="your-api-key"
# Or use Claude, Gemini, or 100+ other providers
```

### Minimal Example
```python
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook

# Initialize with any LLM
client = LiteLLMClient(model="gpt-4o-mini")
generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)
playbook = Playbook()

# Generate with learned strategies
result = generator.generate(
    question="Give me the seahorse emoji",
    context="",
    playbook=playbook
)
print(result.final_answer)
```

---

## Core Concepts
Source: ace/playbook.py, ace/roles.py, ace/delta.py

### Playbook
Stores learned strategies as bullets with helpful/harmful counters.

```python
from ace import Playbook, Bullet

playbook = Playbook()

# Add a strategy
bullet = Bullet(
    content="Always verify emoji existence before returning",
    helpful_count=5,
    harmful_count=1
)
playbook.add_bullet(bullet)

# Get all strategies
strategies = playbook.bullets()

# Convert to prompt format
prompt = playbook.as_prompt()

# Save/load playbook
playbook.save("my_strategies.json")
loaded = Playbook.load("my_strategies.json")
```

### Three Agentic Roles

#### 1. Generator
Executes tasks using playbook strategies.

```python
from ace import Generator, GeneratorOutput

generator = Generator(llm_client)
output: GeneratorOutput = generator.generate(
    question="What is 2+2?",
    context="Mathematics problem",
    playbook=playbook
)
print(output.final_answer)  # "4"
print(output.reasoning)      # Generator's thought process
```

#### 2. Reflector
Analyzes performance and identifies what worked/failed.

```python
from ace import Reflector, ReflectorOutput

reflector = Reflector(llm_client)
reflection: ReflectorOutput = reflector.reflect(
    question="What is 2+2?",
    generator_output=output,
    playbook=playbook,
    ground_truth="4",
    feedback="Correct!"
)
print(reflection.reflection)  # Analysis of performance
print(reflection.helpful_bullets)  # Which strategies helped
print(reflection.harmful_bullets)  # Which strategies hurt
```

#### 3. Curator
Updates playbook based on reflection.

```python
from ace import Curator, CuratorOutput

curator = Curator(llm_client)
curator_output: CuratorOutput = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="math problems",
    progress="improving steadily"
)

# Apply updates to playbook
playbook.apply_delta(curator_output.delta)
```

---

## Complete Examples

### Basic Adaptation
```python
from ace import (
    LiteLLMClient, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment,
    EnvironmentResult, Playbook
)

# Custom environment for task evaluation
class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth in generator_output.final_answer
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth
        )

# Setup ACE
llm = LiteLLMClient(model="gpt-3.5-turbo")
adapter = OfflineAdapter(
    playbook=Playbook(),
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm)
)

# Training samples
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is 5*3?", ground_truth="15"),
]

# Run adaptation
environment = MathEnvironment()
results = adapter.run(samples, environment, epochs=2)

print(f"Learned {len(adapter.playbook.bullets())} strategies")
```

### Self-Reflection Without Ground Truth
```python
# ACE can learn from self-reflection alone
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)
playbook = Playbook()

# Generate answer
output = generator.generate(
    question="Is there a seahorse emoji?",
    context="",
    playbook=playbook
)

# Self-reflect (no ground truth needed)
reflection = reflector.reflect(
    question="Is there a seahorse emoji?",
    generator_output=output,
    playbook=playbook,
    ground_truth=None,  # No ground truth
    feedback=None       # No external feedback
)

# Update playbook
curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="emoji questions",
    progress="self-learning"
)
playbook.apply_delta(curator_output.delta)
```

### Online Adaptation
```python
from ace import OnlineAdapter

# Online learning - adapts during test time
adapter = OnlineAdapter(
    playbook=Playbook(),
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm)
)

# Process test samples sequentially
test_samples = [...]
results = adapter.run(test_samples, environment)
```

---

## 🔧 Advanced Features

### Delta Operations
Source: ace/delta.py

```python
from ace import DeltaOperation, DeltaBatch

# Create delta operations
add_op = DeltaOperation(
    operation="ADD",
    content="New strategy to add"
)

update_op = DeltaOperation(
    operation="UPDATE",
    index=0,
    helpful_delta=1,  # Increment helpful count
    harmful_delta=0
)

remove_op = DeltaOperation(
    operation="REMOVE",
    index=2
)

# Batch operations
batch = DeltaBatch([add_op, update_op, remove_op])
playbook.apply_delta(batch)
```

### Custom Prompts
Source: ace/prompts.py, ace/prompts_v2.py

```python
# Use v2 prompts for better performance
from ace.prompts_v2 import GeneratorPromptV2, ReflectorPromptV2

generator = Generator(
    llm_client,
    prompt_class=GeneratorPromptV2
)

reflector = Reflector(
    llm_client,
    prompt_class=ReflectorPromptV2
)
```

### Explainability
Source: ace/explainability/

```python
from ace.explainability import (
    EvolutionTracker,
    AttributionAnalyzer,
    InteractionTracer,
    ExplainabilityVisualizer
)

# Track playbook evolution
tracker = EvolutionTracker()
tracker.record_state(playbook, step=0)
# ... run adaptation ...
tracker.record_state(playbook, step=1)

# Analyze bullet impact
analyzer = AttributionAnalyzer(tracker)
high_impact = analyzer.get_high_impact_bullets(threshold=0.7)

# Visualize results
visualizer = ExplainabilityVisualizer(tracker)
visualizer.plot_evolution()
visualizer.plot_bullet_impact()
```

---

## LLM Providers
Source: ace/llm_providers/

### OpenAI
```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2000
)
```

### Anthropic Claude
```python
client = LiteLLMClient(
    model="claude-3-5-sonnet-20241022",
    api_key="your-anthropic-key"  # Or set ANTHROPIC_API_KEY
)
```

### Google Gemini
```python
client = LiteLLMClient(
    model="gemini-pro",
    api_key="your-google-key"  # Or set GOOGLE_API_KEY
)
```

### Local Models (Ollama)
```python
client = LiteLLMClient(
    model="ollama/llama2",
    api_base="http://localhost:11434"
)
```

### LangChain Integration
```python
from ace.llm_providers import LangChainClient
from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(model="gpt-4")
client = LangChainClient(langchain_llm)
```

### With Fallbacks
```python
client = LiteLLMClient(
    model="gpt-4",
    fallbacks=["claude-3-haiku", "gpt-3.5-turbo"],
    retry_strategy="exponential_backoff"
)
```

---

## 📊 Benchmarking
Source: benchmarks/

### Run Benchmarks
```python
from benchmarks import BenchmarkManager
from benchmarks.environments import FinerOrdEnvironment

manager = BenchmarkManager()
results = manager.run_benchmark(
    benchmark_name="finer_ord",
    model="gpt-3.5-turbo",
    limit=100,
    epochs=2
)

print(f"Accuracy: {results.accuracy}")
print(f"Improvement: {results.improvement}")
```

### Compare Baseline vs ACE
```python
# scripts/compare_baseline_vs_ace.py
python scripts/compare_baseline_vs_ace.py \
    finer_ord \
    --model gpt-3.5-turbo \
    --samples 50 \
    --epochs 2
```

---

## API Reference

### Core Classes

#### Playbook
```python
class Playbook:
    def __init__(self, bullets: List[Bullet] = None)
    def add_bullet(self, bullet: Bullet) -> None
    def remove_bullet(self, index: int) -> None
    def update_bullet(self, index: int, helpful_delta: int, harmful_delta: int) -> None
    def bullets(self) -> List[Bullet]
    def as_prompt(self) -> str
    def apply_delta(self, delta: Union[DeltaOperation, DeltaBatch]) -> None
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> Playbook
```

#### Generator
```python
class Generator:
    def __init__(self, llm_client: LLMClient, prompt_class=GeneratorPrompt)
    def generate(
        self,
        question: str,
        context: str,
        playbook: Playbook
    ) -> GeneratorOutput
```

#### Reflector
```python
class Reflector:
    def __init__(self, llm_client: LLMClient, prompt_class=ReflectorPrompt)
    def reflect(
        self,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> ReflectorOutput
```

#### Curator
```python
class Curator:
    def __init__(self, llm_client: LLMClient, prompt_class=CuratorPrompt)
    def curate(
        self,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str = "",
        progress: str = ""
    ) -> CuratorOutput
```

#### Adapters
```python
class OfflineAdapter:
    def __init__(
        self,
        playbook: Playbook,
        generator: Generator,
        reflector: Reflector,
        curator: Curator
    )
    def run(
        self,
        samples: List[Sample],
        environment: TaskEnvironment,
        epochs: int = 1
    ) -> List[AdapterStepResult]

class OnlineAdapter:
    # Same interface as OfflineAdapter but processes samples sequentially
```

#### Sample & Environment
```python
@dataclass
class Sample:
    question: str
    ground_truth: Optional[str] = None
    context: str = ""
    metadata: Dict = field(default_factory=dict)

class TaskEnvironment(ABC):
    @abstractmethod
    def evaluate(
        self,
        sample: Sample,
        generator_output: GeneratorOutput
    ) -> EnvironmentResult

@dataclass
class EnvironmentResult:
    feedback: str
    ground_truth: Optional[str] = None
    success: bool = False
    metadata: Dict = field(default_factory=dict)
```

---

## Common Tasks

### Task: Create a self-improving chatbot
```python
# The agent learns from user feedback
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook

llm = LiteLLMClient(model="gpt-3.5-turbo")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

def chat_with_learning(user_input: str, user_feedback: str = None):
    # Generate response
    output = generator.generate(
        question=user_input,
        context="chatbot conversation",
        playbook=playbook
    )

    # If user provides feedback, learn from it
    if user_feedback:
        reflection = reflector.reflect(
            question=user_input,
            generator_output=output,
            playbook=playbook,
            feedback=user_feedback
        )
        curator_output = curator.curate(
            reflection=reflection,
            playbook=playbook
        )
        playbook.apply_delta(curator_output.delta)

    return output.final_answer
```

### Task: Benchmark ACE performance
```python
python scripts/run_benchmark.py finer_ord \
    --model gpt-4o-mini \
    --limit 100 \
    --epochs 3 \
    --save-detailed
```

### Task: Persist playbook across sessions
```python
from ace import Playbook

# Save after training
playbook.save("learned_strategies.json")

# Load in new session
playbook = Playbook.load("learned_strategies.json")

# Continue with existing knowledge
generator = Generator(llm)
output = generator.generate(
    question="New question",
    context="",
    playbook=playbook  # Uses previously learned strategies
)
```

### Task: Explain what ACE learned
```python
from ace.explainability import AttributionAnalyzer, ExplainabilityVisualizer

# After running adaptation
analyzer = AttributionAnalyzer(evolution_tracker)
visualizer = ExplainabilityVisualizer(evolution_tracker)

# Get most impactful strategies
top_bullets = analyzer.get_high_impact_bullets(threshold=0.8)
for bullet in top_bullets:
    print(f"Strategy: {bullet.content}")
    print(f"Impact: {bullet.impact_score}")

# Visualize learning process
visualizer.plot_evolution(save_path="learning_curve.png")
visualizer.plot_bullet_impact(save_path="strategy_impact.png")
```

---

## Troubleshooting

### API Key Issues
```python
# Option 1: Environment variable
export OPENAI_API_KEY="sk-..."

# Option 2: Pass directly
client = LiteLLMClient(
    model="gpt-4",
    api_key="sk-..."
)

# Option 3: Use .env file
from dotenv import load_dotenv
load_dotenv()
```

### Memory/Context Issues
```python
# Limit playbook size
playbook.truncate(max_bullets=50)

# Use smaller context windows
generator = Generator(
    llm_client,
    max_context_length=2000
)
```

### Performance Optimization
```python
# Use caching
client = LiteLLMClient(
    model="gpt-3.5-turbo",
    cache=True
)

# Batch processing
adapter = OfflineAdapter(...)
results = adapter.run(
    samples,
    environment,
    batch_size=10  # Process 10 at a time
)
```

---

## Examples Directory

Full working examples available at `examples/`:

- `simple_ace_example.py` - Minimal ACE setup
- `kayba_ace_test.py` - The famous seahorse emoji demo
- `quickstart_litellm.py` - Production LLM integration
- `langchain_example.py` - LangChain integration
- `playbook_persistence.py` - Save/load strategies
- `compare_v1_v2_prompts.py` - Prompt version comparison
- `advanced_prompts_v2.py` - Enhanced prompt usage

---

## Next Steps

1. **Install ACE**: `pip install ace-framework`
2. **Run the Kayba Test**: `python examples/kayba_ace_test.py`
3. **Read the paper**: [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
4. **Join Discord**: [discord.gg/mqCqH7sTyK](https://discord.gg/mqCqH7sTyK)

---

*Built with ❤️ by [Kayba AI](https://kayba.ai) and the open-source community*
