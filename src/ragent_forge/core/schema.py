from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

WORKSPACE_SCHEMA_VERSION = 2

SchemaRecord = dict[str, Any]
SchemaMigration = Callable[[SchemaRecord], SchemaRecord]


@dataclass(frozen=True)
class SchemaMigrationStep:
    source_version: int
    target_version: int
    name: str


@dataclass(frozen=True)
class SchemaMigrationPlan:
    artifact: str
    source_version: int
    target_version: int
    steps: tuple[SchemaMigrationStep, ...]

    @property
    def required(self) -> bool:
        return bool(self.steps)


def _legacy_to_v1(record: SchemaRecord) -> SchemaRecord:
    return {**record, "schema_version": 1}


def _v1_to_v2(record: SchemaRecord) -> SchemaRecord:
    return {**record, "schema_version": 2}


_MIGRATIONS: dict[int, tuple[str, SchemaMigration]] = {
    0: ("legacy_to_v1", _legacy_to_v1),
    1: ("v1_to_v2", _v1_to_v2),
}


def add_schema_version(record: SchemaRecord) -> SchemaRecord:
    return {
        **record,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
    }


def schema_migration_plan(
    record: SchemaRecord,
    artifact: str,
) -> SchemaMigrationPlan:
    source_version = _schema_version(record, artifact)
    steps: list[SchemaMigrationStep] = []
    version = source_version
    while version < WORKSPACE_SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(
                f"Unsupported {artifact} schema_version {version}; no migration "
                f"to version {version + 1} is registered"
            )
        name, _migrate = migration
        steps.append(
            SchemaMigrationStep(
                source_version=version,
                target_version=version + 1,
                name=name,
            )
        )
        version += 1
    return SchemaMigrationPlan(
        artifact=artifact,
        source_version=source_version,
        target_version=WORKSPACE_SCHEMA_VERSION,
        steps=tuple(steps),
    )


def migrate_schema_record(
    record: SchemaRecord,
    artifact: str,
) -> SchemaRecord:
    plan = schema_migration_plan(record, artifact)
    migrated = dict(record)
    for step in plan.steps:
        _name, migrate = _MIGRATIONS[step.source_version]
        migrated = migrate(migrated)
    return migrated


def validate_schema_version(record: SchemaRecord, artifact: str) -> None:
    schema_migration_plan(record, artifact)


def _schema_version(record: SchemaRecord, artifact: str) -> int:
    version = record.get("schema_version", 0)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"Invalid {artifact}: schema_version must be an integer")
    if version < 0:
        raise ValueError(
            f"Unsupported {artifact} schema_version {version}; minimum supported "
            "version is 0"
        )
    if version > WORKSPACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {artifact} schema_version {version}; "
            f"maximum supported version is {WORKSPACE_SCHEMA_VERSION}"
        )
    return version
