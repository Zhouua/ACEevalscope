# Changelog

All notable changes to ACE Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-16

### Added
- **Experimental v2 Prompts** with state-of-the-art prompt engineering
  - Confidence scoring at bullet and answer levels
  - Domain-specific variants for math and code generation
  - Hierarchical structure with identity headers and metadata
  - Concrete examples and anti-patterns for better guidance
  - PromptManager for version control and A/B testing
- Comprehensive prompt engineering documentation (`docs/PROMPT_ENGINEERING.md`)
- Advanced examples demonstrating v2 prompts (`examples/advanced_prompts_v2.py`)
- Comparison script for v1 vs v2 prompts (`examples/compare_v1_v2_prompts.py`)
- Playbook persistence with `save_to_file()` and `load_from_file()` methods
- Example demonstrating playbook save/load functionality (`examples/playbook_persistence.py`)
- py.typed file for PEP 561 type hint support
- Mermaid flowchart visualization in README showing ACE learning loop
- ACE_ROADMAP.md (untracked) for development planning

### Changed
- Enhanced docstrings with comprehensive examples throughout codebase
- Improved README with v2 prompts section and visual diagrams
- Updated formatting to comply with Black code style

### Fixed
- README incorrectly referenced non-existent docs/ directory
- Test badge URL in README (test.yml → tests.yml)
- Code formatting issues detected by GitHub Actions

## [0.2.0] - 2025-10-15

### Added
- LangChain integration via `LangChainLiteLLMClient` for advanced workflows
- Router support for load balancing across multiple model deployments
- Comprehensive example for LangChain usage (`examples/langchain_example.py`)
- Optional installation group: `pip install ace-framework[langchain]`
- PyPI badges and Quick Links section in README
- CHANGELOG.md for version tracking

### Fixed
- Parameter filtering in LiteLLM and LangChain clients (refinement_round, max_refinement_rounds)
- GitHub Actions workflow using deprecated artifact actions v3 → v4

### Changed
- Improved README with better structure and badges
- Updated .gitignore to exclude build artifacts and development files

### Removed
- Unnecessary development files from repository

## [0.1.1] - 2025-10-15

### Fixed
- GitHub Actions workflow for PyPI publishing
- Updated artifact upload/download actions from v3 to v4

## [0.1.0] - 2025-10-15

### Added
- Initial release of ACE Framework
- Core ACE implementation based on paper (arXiv:2510.04618)
- Three-role architecture: Generator, Reflector, and Curator
- Playbook system for storing and evolving strategies
- LiteLLM integration supporting 100+ LLM providers
- Offline and Online adaptation modes
- Async and streaming support
- Example scripts for quick start
- Comprehensive test suite
- PyPI packaging and GitHub Actions CI/CD

### Features
- Self-improving agents that learn from experience
- Delta operations for incremental playbook updates
- Support for OpenAI, Anthropic, Google, and more via LiteLLM
- Type hints and modern Python practices
- MIT licensed for open source use

[0.2.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kayba-ai/agentic-context-engine/releases/tag/v0.1.0