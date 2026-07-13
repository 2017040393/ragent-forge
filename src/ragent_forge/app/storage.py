"""Compatibility imports for local storage helpers."""

from ragent_forge.infrastructure.storage import (
    atomic_write_text,
    workspace_write_lock,
)

__all__ = ["atomic_write_text", "workspace_write_lock"]
