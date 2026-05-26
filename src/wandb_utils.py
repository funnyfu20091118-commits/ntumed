"""
W&B helpers for consistent logging across stages.
"""
import time
from typing import Any, Dict, Optional

try:
    import wandb
except Exception:
    wandb = None


def init_wandb(cfg, stage: str, run_type: str = ""):
    if wandb is None:
        print("wandb not installed; skipping logging.")
        return None
    if not getattr(cfg, "wandb_enabled", True):
        print("wandb disabled; skipping logging.")
        return None
    project = getattr(cfg, "wandb_project", None)
    if not project:
        print("wandb project not set; skipping logging.")
        return None
    entity = getattr(cfg, "wandb_entity", None) or None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stage}-{timestamp}" if getattr(cfg, "wandb_auto_name", True) else None
    tags = [stage]
    if run_type:
        tags.append(run_type)
    settings = wandb.Settings(start_method="thread")
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        config=vars(cfg),
        settings=settings,
    )


def log_metrics(run, metrics: Dict[str, Any], step: Optional[int] = None):
    if run is None:
        return
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)
