"""Shared settings used by every module (change values here, not inside the other files)."""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "SeoulBikeData.csv")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
FIG_DIR    = os.path.join(OUT_DIR, "figures")

SEED       = 42      # makes every random step reproducible
TEST_SIZE  = 0.30    # 70% train / 30% test
ALPHA      = 0.05    # significance level for all hypothesis tests
N_BOOT     = 10000   # bootstrap resamples

os.makedirs(FIG_DIR, exist_ok=True)
