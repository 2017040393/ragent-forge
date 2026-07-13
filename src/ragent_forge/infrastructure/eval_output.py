from __future__ import annotations

import json
from pathlib import Path

from ragent_forge.app.services.eval_dataset_generation_service import (
    GeneratedEvalCase,
)
from ragent_forge.infrastructure.storage import atomic_write_text


def write_generated_eval_jsonl(
    cases: list[GeneratedEvalCase],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output JSONL already exists: {path}")

    lines = [
        json.dumps(case.to_jsonl_record(), ensure_ascii=False, sort_keys=True)
        for case in cases
    ]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return path
