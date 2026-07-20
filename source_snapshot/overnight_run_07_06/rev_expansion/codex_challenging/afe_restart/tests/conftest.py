"""Keep the isolated restart package importable under the repository pytest root."""
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

