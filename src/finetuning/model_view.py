"""A view of the base checkpoint whose processor uses the configured image token budget.

Training must see the same image tokens as extraction. The extraction backend passes
mm_processor_kwargs at run time; the training tool reads processor_config.json, so the
view is a directory of symlinks to the checkpoint with only that file rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_logging import log_step_start


def build_model_view(context, *, model, output):
    source = context.model_path(model)
    budget = (context.model_settings(model).get("mm_processor_kwargs") or {}).get("max_soft_tokens")
    target = Path(output).expanduser().resolve()
    log_step_start(context, "model-view", model=model, source=source, output=target,
                   max_soft_tokens=budget)
    if budget is None:
        raise ValueError(f"models.{model}.mm_processor_kwargs.max_soft_tokens is not set")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.name.startswith(".") or path.name == "processor_config.json":
            continue
        (target / path.name).symlink_to(path)
    config = json.loads((source / "processor_config.json").read_text(encoding="utf-8"))
    config["image_processor"]["max_soft_tokens"] = budget
    config["image_processor"]["image_seq_length"] = budget
    config["image_seq_length"] = budget
    (target / "processor_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[MODEL-VIEW] {target} (max_soft_tokens={budget})", flush=True)
    return {"failure_count": 0}
