"""Dependency provenance for the clean AFE restart.

The legacy workspace contains several same-named experiment modules.  This
module makes their resolved locations explicit and records content hashes in
every run directory.  It intentionally never imports either legacy expansion
trainer.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[2]

FORBIDDEN_MODULE_FRAGMENTS = (
    "grid_expand_hardtail",
    "window_expand_hardtail",
    "stage5_window_expand",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_provenance(module: ModuleType) -> dict[str, str]:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"module {module.__name__!r} has no source file")
    path = Path(raw).resolve()
    return {
        "module": module.__name__,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def assert_no_legacy_expansion_imports(module_names: Iterable[str] | None = None) -> None:
    """Fail if a clean-restart process imported a legacy expansion trainer."""
    if module_names is None:
        import sys

        module_names = tuple(sys.modules)
    offenders = sorted(
        name for name in module_names
        if any(fragment in name for fragment in FORBIDDEN_MODULE_FRAGMENTS)
    )
    if offenders:
        raise RuntimeError(f"legacy expansion modules imported: {offenders}")


def resolve_core_dependencies() -> dict[str, dict[str, str]]:
    """Resolve and hash the deliberately reused architecture/scene modules."""
    names = (
        "grid_hp_expt",
        "grid_feats",
        "grid_scene",
        "flow_policy",
        "verifier_polytope",
        "cfm_mppi.safegpc_adapter.safemppi",
    )
    resolved = {}
    for name in names:
        module = importlib.import_module(name)
        resolved[name] = module_provenance(module)
    assert_no_legacy_expansion_imports()
    return resolved


def write_dependency_manifest(output: str | Path) -> dict[str, object]:
    output = Path(output)
    payload: dict[str, object] = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "project_root": str(PROJECT_ROOT),
        "dependencies": resolve_core_dependencies(),
        "forbidden_module_fragments": list(FORBIDDEN_MODULE_FRAGMENTS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
