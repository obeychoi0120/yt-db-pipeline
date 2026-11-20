"""Pick the best LoRA checkpoint of a run and merge it into a standalone model."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pipeline_logging import log_step_start


def find_best_checkpoint(training_dir: Path) -> tuple[Path, float | None]:
    """The checkpoint with the lowest eval_loss in the newest ms-swift run directory."""
    runs = sorted(
        (path for path in training_dir.glob("v*-*") if any(path.glob("checkpoint-*"))),
        key=lambda path: int(re.match(r"v(\d+)-", path.name).group(1)),
    )
    if not runs:
        raise ValueError(f"no ms-swift run with checkpoints under {training_dir}")
    run = runs[-1]
    checkpoints = sorted(run.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]))
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    by_step = {path.name.split("-")[1]: path for path in checkpoints}
    losses = [
        (entry["eval_loss"], by_step[str(entry["step"])])
        for entry in state.get("log_history", [])
        if "eval_loss" in entry and str(entry.get("step")) in by_step
    ]
    if not losses:
        return checkpoints[-1], None
    loss, checkpoint = min(losses, key=lambda item: item[0])
    return checkpoint, loss


def merge_best(context, *, training_dir, output):
    training_dir = Path(training_dir).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    checkpoint, loss = find_best_checkpoint(training_dir)
    log_step_start(context, "merge", checkpoint=checkpoint, eval_loss=loss, output=target)
    if target.exists():
        raise ValueError(f"output already exists; choose a new directory: {target}")
    subprocess.run(
        ["swift", "export", "--adapters", str(checkpoint), "--merge_lora", "true",
         "--output_dir", str(target)],
        check=True,
    )
    print(f"[MERGE] {target}\nAdd it to config models with backend: vllm to extract with it.",
          flush=True)
    return {"failure_count": 0}
