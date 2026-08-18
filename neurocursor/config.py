"""Project-wide constants. No magic numbers scattered across modules."""

from pathlib import Path

# Reproducibility
SEED = 42

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# Primary real dataset: O'Doherty et al. 2017 (Zenodo record 583331)
ZENODO_RECORD = "583331"
DEFAULT_SESSION_FILE = "indy_20161005_06.mat"
DEFAULT_SESSION_MD5 = "5ea300952642e0fc54245144499db9bb"
ZENODO_FILE_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    f"/files/{DEFAULT_SESSION_FILE}/content"
)

# Binning / preprocessing
BIN_MS = 64.0  # bin duration in milliseconds (must be > per-bin decode latency)
BIN_S = BIN_MS / 1000.0
WIENER_TAPS = 4  # number of neural history bins (current + past) for the Wiener/ridge decoder

# Temporal split (leakage-safe: earlier time = train, later time = test)
TRAIN_FRACTION = 0.75

# Task-success proxy (open-loop offline integration of decoded velocity)
# Acceptance radius is expressed as a fraction of the median reach distance so
# it scales with the workspace; the absolute mm value is reported in results.
ACCEPT_RADIUS_FRACTION = 0.25
MIN_REACH_DISTANCE_MM = 5.0  # ignore micro-reaches below this start->target distance

# Ridge regularization search grid
RIDGE_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
