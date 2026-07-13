from __future__ import annotations

from typing import Any

WORKSPACE_SCHEMA_VERSION = 1


def add_schema_version(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
    }


def validate_schema_version(record: dict[str, Any], artifact: str) -> None:
    version = record.get("schema_version")
    if version is None:
        # Files written before schema versioning remain readable.
        return
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"Invalid {artifact}: schema_version must be an integer")
    if version > WORKSPACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {artifact} schema_version {version}; "
            f"maximum supported version is {WORKSPACE_SCHEMA_VERSION}"
        )
