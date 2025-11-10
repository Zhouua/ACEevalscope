"""
Lightweight benchmark integration framework for ACE.

This module provides configuration-driven benchmark evaluation that follows
production patterns from lm-evaluation-harness, keeping repositories lightweight
while supporting comprehensive evaluation capabilities.

Key Features:
- YAML-based task configuration
- Pluggable data loaders for different sources
- Automatic caching and streaming
- Integration with ACE's TaskEnvironment pattern

Usage:
    >>> from benchmarks import BenchmarkTaskManager
    >>> manager = BenchmarkTaskManager()
    >>> benchmark = manager.get_benchmark("finer")
    >>> results = benchmark.run_evaluation(generator, samples)
"""

from .base import (
    BenchmarkConfig,
    BenchmarkEnvironment,
    BenchmarkSample,
    DataLoader,
    get_cache_dir,
    get_data_dir,
)
from .manager import BenchmarkTaskManager

__all__ = [
    "BenchmarkConfig",
    "BenchmarkEnvironment",
    "BenchmarkSample",
    "BenchmarkTaskManager",
    "DataLoader",
    "get_cache_dir",
    "get_data_dir",
]

__version__ = "0.1.0"