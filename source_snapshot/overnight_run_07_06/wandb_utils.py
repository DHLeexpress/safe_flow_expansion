"""Weights & Biases helper (online by default now that the user logged in).

Only numeric values are logged as metrics; images/videos via log_image/log_video. Graceful degradation
if wandb is missing/broken (the stage still runs).
"""
from __future__ import annotations

import os


def add_wandb_args(ap, default_project="cfm-mppi-safeflow"):
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default=default_project)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="narrow-gap")
    return ap


def init_run(args, name, config, dir=None, group=None):
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        return wandb.init(project=getattr(args, "wandb_project", "cfm-mppi-safeflow"),
                          name=getattr(args, "wandb_name", None) or name,
                          group=group or getattr(args, "wandb_group", None),
                          mode=args.wandb_mode, dir=dir, config=config, reinit=True)
    except Exception as exc:
        print(f"[wandb] disabled ({exc})", flush=True)
        return None


def log(run, data, step=None):
    if run is not None:
        run.log({k: v for k, v in data.items() if isinstance(v, (int, float, bool))}, step=step)


def log_image(run, key, path):
    if run is not None and path and os.path.exists(path):
        try:
            import wandb
            run.log({key: wandb.Image(path)})
        except Exception as exc:
            print(f"[wandb] image log failed for {key} ({exc})", flush=True)


def log_video(run, key, path):
    if run is not None and path and os.path.exists(path):
        try:
            import wandb
            run.log({key: wandb.Video(path)})
        except Exception as exc:
            print(f"[wandb] video log failed for {key} ({exc})", flush=True)


def finish(run, summary=None):
    if run is not None:
        try:
            for k, v in (summary or {}).items():
                run.summary[k] = v
            run.finish()
        except Exception as exc:
            print(f"[wandb] finish failed ({exc})", flush=True)
