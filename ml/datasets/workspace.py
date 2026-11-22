"""A dataset workspace: one dataset's history, replay stream, config, registry and logs, kept apart from the rest."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "ml" / "workspaces"


@dataclass(frozen=True)
class Workspace:
    name: str
    root: Path = DEFAULT_ROOT

    @property
    def path(self) -> Path:
        return Path(self.root) / self.name

    raw_path = property(lambda self: self.path / "raw_transactions.csv")
    stream_path = property(lambda self: self.path / "stream.csv")
    data_dir = property(lambda self: self.path / "data_store")
    registry_dir = property(lambda self: self.path / "registry_store")
    config_path = property(lambda self: self.path / "config.json")

    def exists(self) -> bool:
        return self.raw_path.exists() and self.config_path.exists()

    def config(self) -> dict:
        if not self.exists():
            raise FileNotFoundError(f"No prepared workspace {self.name!r} under {self.root}. "
                                    f"Run: python -m ml.datasets.prepare {self.name} --path <folder>")
        return json.loads(self.config_path.read_text())
