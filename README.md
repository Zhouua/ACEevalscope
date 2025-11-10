# ACEevalscope
实现将ACE的代码接入Evalscope进行完整测评，支持多种benchmark，快速测评

## Link
<a href="https://arxiv.org/pdf/2510.04618">Paper: Agentic-Context-Engineering</a><br>
<a href="https://github.com/kayba-ai/agentic-context-engine">Agentic-Context-Engineering Framework</a><br>
<a href="https://github.com/modelscope/evalscope">Evalscope</a><br>

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/kayba-ai/ACEevalscope.git
   cd ACEevalscope
   ```
2. Install dependencies:
   ```bash
   conda create -n aceevalscope python=3.12
   conda activate aceevalscope
   cd ACEevalscope/agentic-context-engine
   pip install ace-framework
   cd ACEevalscope/evalscope
   pip install -e .
   ```
3. Set up the environment:
   ```bash
   cd ACEevalscope
   cp .env.example .env
   ```
4. Run the test:
   ```bash
   cd ACEevalscope/evalscope
   python -m unittest tests.benchmark.test_eval.TestNativeBenchmark.test_gsm8k
   ```
5. To run the benchmark:
- You only need to change the ACEevalscope/evalscope/evalscope/benchmarks/xxx/xxx_adapter.py file to use ACEAdapter.
- Then run the evaluation as the evalscope would<br>
example:
   ```bash
   # ACEevalscope/evalscope/evalscope/benchmarks/aime/aime24_adapter.py
    class AIME24Adapter(DefaultAdapter):
    # Change to:
    class AIME24Adapter(ACEAdapter):

        def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'problem_id': record.get('id', ''),
                'solution': record.get('solution', ''),
            },
        )
   ```


