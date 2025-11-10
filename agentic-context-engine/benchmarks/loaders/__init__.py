"""Data loaders for different benchmark sources."""

from .huggingface import HuggingFaceLoader

# AppWorld loader is imported conditionally since appworld might not be installed
try:
    from .appworld import AppWorldLoader
    __all__ = ["HuggingFaceLoader", "AppWorldLoader"]
except ImportError:
    __all__ = ["HuggingFaceLoader"]