from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from visual_sampling import validate_sampling

# Relative to the repository root; YT_DB_PIPELINE_CONFIG selects another file.
CONFIG_PATH = Path(os.environ.get("YT_DB_PIPELINE_CONFIG") or "config.yaml")
# Generation backends an extraction.models entry may name.
GENERATION_BACKENDS = ("vllm", "gemini")


class ConfigError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be an object: {path}")
    return value


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class RunContext:
    """Paths and settings of one invocation.

    Preparation is shared by every Run and needs no run_id; extraction, scoring,
    fine-tuning and evaluation write under ``artifacts/runs/<run_id>``.
    """

    root: Path
    run_id: str | None
    config: dict[str, Any]
    artifact_root: Path

    @classmethod
    def load(cls, run_id: str | None = None, *, root: Path | None = None) -> "RunContext":
        repo_root = (root or Path(__file__).resolve().parents[1]).resolve()
        selected = None
        if run_id is not None:
            selected = str(run_id).strip()
            if (
                not selected
                or selected in {".", ".."}
                or Path(selected).name != selected
                or ("\\" in selected)
            ):
                raise ConfigError("run_id must be a single non-empty directory name")
        config = _load_yaml(repo_root / CONFIG_PATH)
        _validate_config(config)
        artifact_root = _resolve(
            repo_root, config.get("artifacts_root", "artifacts"), "artifacts_root"
        )
        return cls(repo_root, selected, config, artifact_root)

    @property
    def run_root(self) -> Path:
        if self.run_id is None:
            raise ConfigError("this step requires --run-id")
        return self.artifact_root / "runs" / self.run_id

    # Preparation (shared by every Run)

    @property
    def preparation_dir(self) -> Path:
        return self.artifact_root / "preparation"

    @property
    def manifest_path(self) -> Path:
        return self.preparation_dir / "manifest.csv"

    @property
    def splits_path(self) -> Path:
        return self.preparation_dir / "splits.csv"

    @property
    def data_root(self) -> Path:
        """Downloaded videos and metadata; set by preparation.data_root."""
        preparation = _require_mapping(self.config, "preparation")
        return _resolve(self.root, preparation.get("data_root"), "preparation.data_root")

    @property
    def videos_dir(self) -> Path:
        return self.data_root / "videos"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "metadata"

    @property
    def keyframes_dir(self) -> Path:
        width, height = self.config["extraction"]["visual_evidence"]["image_resolution"]
        return self.preparation_dir / "resized_keyframes" / f"{width}_{height}"

    @property
    def source_assets_dir(self) -> Path:
        return self.preparation_dir / "source_assets"

    # Run outputs

    def scene_dir(self, model: str) -> Path:
        return self.run_root / "extraction" / scene_arm(model)

    def scoring_dir(self, model: str) -> Path:
        return self.run_root / "scoring" / scene_arm(model)

    @property
    def iab_dir(self) -> Path:
        """IAB taxonomy labels of each scene (extract-iab-gt)."""
        return self.run_root / "extraction" / "iab_gt"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    def prompt_path(self, schema: str | Path) -> Path:
        path = _resolve(self.root, str(schema), "--schema")
        if not path.is_file():
            raise ValueError(f"--schema prompt not found: {path}")
        return path

    # Models

    def model_settings(self, model: str) -> dict[str, Any]:
        """One generation model (a --model name) of extraction.models."""
        models = self.config["extraction"].get("models")
        value = models.get(model) if isinstance(models, dict) else None
        if isinstance(value, str):
            value = {"path": value}
        if not isinstance(value, dict):
            raise ConfigError(f"extraction.models.{model} is missing; add it to {CONFIG_PATH}")
        return value

    def model_path(self, model: str) -> Path:
        return _resolve(
            self.root, self.model_settings(model).get("path"), f"extraction.models.{model}.path"
        )

    def step_path(self, step: str, key: str) -> Path:
        """A model path owned by one step, such as scoring.d3_path or evaluation.bge_path."""
        return _resolve(self.root, (self.config.get(step) or {}).get(key), f"{step}.{key}")

    def generation_backend(self, model: str) -> str:
        backend = self.model_settings(model).get("backend")
        if backend not in GENERATION_BACKENDS:
            raise ConfigError(
                f"extraction.models.{model}.backend must be one of {', '.join(GENERATION_BACKENDS)}"
            )
        return backend

    def max_new_tokens(self, model: str) -> int:
        value = self.model_settings(model).get("scene_max_new_tokens")
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"extraction.models.{model}.scene_max_new_tokens must be a positive integer"
            )
        return value


def scene_arm(model: str) -> str:
    """Directory name of one model's scene graphs."""
    return f"graph_{model}"


def _validate_config(value: dict[str, Any]) -> None:
    """Only check what every step needs to start; everything else is read where it is used."""
    if "models" in value:
        raise ConfigError(
            "the top-level models section moved into the steps: generation models to "
            "extraction.models, models.d3.path to scoring.d3_path and models.bge.path to "
            "evaluation.bge_path (see config.example.yaml)"
        )
    _require_mapping(value, "extraction")
    visual_evidence = _require_mapping(value["extraction"], "visual_evidence")
    try:
        validate_sampling(visual_evidence["scene_duration"], visual_evidence["num_keyframes"])
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"extraction.visual_evidence: {exc}") from exc
