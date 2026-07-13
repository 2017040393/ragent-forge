"""Compatibility import for the local workspace adapter.

Application code should depend on workspace ports; composition code may use the
concrete adapter from ``ragent_forge.infrastructure.local_workspace``.
"""

from ragent_forge.infrastructure.local_workspace import LocalWorkspace

__all__ = ["LocalWorkspace"]
