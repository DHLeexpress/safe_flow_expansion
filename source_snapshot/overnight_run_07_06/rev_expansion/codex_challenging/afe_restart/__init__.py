"""Clean planned-window AFE restart.

This package intentionally does not import the legacy expansion trainer.  The
only reusable pieces are the scene, policy architecture, deterministic
full-window verifier, and SafeMPPI proposal machinery.
"""

from pathlib import Path
import sys


# The experiment predates packaging and its reusable scene/feature modules
# live two directories above this package.  Bootstrap those locations once,
# then restore this experiment root to highest priority so same-named legacy
# variants cannot shadow the audited local architecture.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_EXPERIMENT_ROOT = _PACKAGE_ROOT.parent
_WORK_ROOT = _EXPERIMENT_ROOT.parents[1]
for _path in (_WORK_ROOT, _EXPERIMENT_ROOT.parent, _EXPERIMENT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import _paths as _shared_paths  # noqa: E402,F401
if str(_EXPERIMENT_ROOT) in sys.path:
    sys.path.remove(str(_EXPERIMENT_ROOT))
sys.path.insert(0, str(_EXPERIMENT_ROOT))
