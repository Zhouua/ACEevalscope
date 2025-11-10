# Development Scripts

⚠️ **Note**: These scripts are for development and research purposes only. They are not included in the PyPI package and may reference local model weights or data files that are not part of the public repository.

## Scripts

- `run_questions.py` - Run ACE adaptation on sample questions with a local model
- `run_questions_direct.py` - Run questions directly without adaptation for baseline comparison
- `run_local_adapter.py` - Test ACE with local Transformers models

## Requirements

These scripts require:
- Local model weights (not included in repo)
- CUDA-capable GPU for local model inference
- Additional dependencies beyond the base package

## Usage

These are primarily for ACE framework development and testing. For production use, please refer to the examples in the `examples/` directory which use cloud LLM providers.