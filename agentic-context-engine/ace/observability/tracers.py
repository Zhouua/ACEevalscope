"""
ACE-specific tracing utilities for Opik integration.

Provides utilities for conditionally applying Opik tracing to ACE framework components.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Check if Opik is enabled via environment variable (default: True if available)
_OPIK_ENABLED = os.getenv('OPIK_ENABLED', 'true').lower() in ('true', '1', 'yes')

# Flag to check if Opik is available
_OPIK_AVAILABLE = False
if _OPIK_ENABLED:
    try:
        import opik
        from opik import track
        _OPIK_AVAILABLE = True
    except ImportError:
        pass


def maybe_track(
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    **kwargs: Any
) -> Callable[[F], F]:
    """
    Conditionally apply @opik.track decorator if Opik is available.

    Args:
        name: Name for the trace
        tags: Tags for the trace
        **kwargs: Additional arguments for @track
    """
    def decorator(func: F) -> F:
        if not _OPIK_AVAILABLE:
            return func

        try:
            # Apply Opik's @track decorator directly
            return track(name=name, tags=tags, **kwargs)(func)
        except Exception as e:
            logger.warning(f"Failed to apply Opik tracking to {func.__name__}: {e}")
            return func

    return decorator


# Legacy aliases for backward compatibility
def track_role(*args, **kwargs):
    """Legacy alias - use maybe_track instead."""
    return maybe_track(*args, **kwargs)


def ace_track(*args, **kwargs):
    """Legacy alias - use maybe_track instead."""
    return maybe_track(*args, **kwargs)