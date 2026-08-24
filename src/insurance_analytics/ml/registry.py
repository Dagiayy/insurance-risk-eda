"""A small, filesystem-based model registry.

The original ``modeling.py`` never persisted a single trained model -- it trained,
scored, and discarded them in one process, so "deploying" a model meant re-running the
whole script. This gives every registered model a stable, versioned home:

    models/<model_name>/<version>/model.joblib
    models/<model_name>/<version>/metadata.json
    models/<model_name>/latest -> <version>          (pointer file, not a symlink -- Windows-safe)

This is intentionally not MLflow: there's no tracking server to run, no extra service to
deploy, and a single-node filesystem registry is the right amount of ceremony for a
project this size (see the "don't overengineer" mandate). It stores everything MLflow's
model registry would need (artifact, metrics, params, timestamp, git-ish version string)
so swapping in a real tracking server later is a drop-in change, not a rewrite.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelMetadata:
    model_name: str
    version: str
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    framework: str = "scikit-learn"
    notes: str = ""


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model_name: str, model: object, metadata: ModelMetadata) -> Path:
        version_dir = self.root / model_name / metadata.version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_dir / "model.joblib")
        (version_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))
        (self.root / model_name / "latest.txt").write_text(metadata.version)
        return version_dir

    def load(self, model_name: str, version: str | None = None) -> tuple[object, ModelMetadata]:
        version = version or self._latest_version(model_name)
        version_dir = self.root / model_name / version
        if not version_dir.exists():
            raise FileNotFoundError(f"No such model version: {model_name}/{version}")

        model = joblib.load(version_dir / "model.joblib")
        meta_dict = json.loads((version_dir / "metadata.json").read_text())
        return model, ModelMetadata(**meta_dict)

    def _latest_version(self, model_name: str) -> str:
        pointer = self.root / model_name / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"No registered versions for model '{model_name}'")
        return pointer.read_text().strip()

    def list_versions(self, model_name: str) -> list[str]:
        model_dir = self.root / model_name
        if not model_dir.exists():
            return []
        return sorted(p.name for p in model_dir.iterdir() if p.is_dir())

    def has_model(self, model_name: str) -> bool:
        return (self.root / model_name / "latest.txt").exists()


def new_version_string() -> str:
    return time.strftime("%Y%m%dT%H%M%S")
