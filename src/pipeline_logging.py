"""Console-only settings for each selected step."""

from __future__ import annotations
import json
from pathlib import Path
from pipeline_runtime import CONFIG_PATH


def log_step_start(context, step, **options):
    values = {"config_file": context.root / CONFIG_PATH}
    if context.run_id is not None:
        values["run_id"] = context.run_id
    values.update(options)
    print(f"[STEP] {step}", flush=True)
    for key, value in values.items():
        print(
            f"  {key}: {(str(value) if isinstance(value, (str, Path)) else json.dumps(value))}",
            flush=True,
        )
