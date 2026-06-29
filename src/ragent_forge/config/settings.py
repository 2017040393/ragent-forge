from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    workspace_path: Path = Path.cwd()
    local_state_dir: str = ".ragent"
