"""Reproducibility seed management — Bug Fix #5.

Sets seeds for torch, numpy, random, and PYTHONHASHSEED to guarantee
deterministic simulation runs.
"""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
