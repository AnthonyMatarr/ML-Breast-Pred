import os

os.environ["OMP_NUM_THREADS"] = "1"  # for torch/sklearn MacOS conflict
from pathlib import Path

SEED = 42424242
BASE_PATH = Path("your_path_here")
DEVICE = "cpu"
