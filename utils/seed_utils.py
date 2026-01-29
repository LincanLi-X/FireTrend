"""
seed_utils.py
------------------------------------
Randomness control utilities for FireTrend project.

Ensures reproducibility across:
- Python random module
- NumPy
- PyTorch (CPU and CUDA)
- DataLoader subprocess workers

Usage:
    from utils.seed_utils import set_seed, seed_worker
    set_seed(42)
    DataLoader(..., worker_init_fn=seed_worker)
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for full reproducibility.

    Args:
        seed (int): Random seed value.
        deterministic (bool): Whether to enforce deterministic cuDNN.
                              Set False for faster training (non-deterministic).
    """

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Control PyTorch backend behavior
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic  # benchmark can speed up when deterministic=False

    # Ensure reproducible hash ordering
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: make torch.use_deterministic_algorithms active (>=1.8)
    # try:
    #     torch.use_deterministic_algorithms(deterministic)
    # except Exception:
    #     pass

    print(f"🌱 Random seed set to {seed} | Deterministic mode = {deterministic}")


def seed_worker(worker_id: int) -> None:
    """
    Ensure each DataLoader worker has a different, yet reproducible, seed.

    Args:
        worker_id (int): Worker process ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_seed() -> int:
    """
    Generate a pseudo-random seed (e.g. for logging unique runs).
    """
    import time
    new_seed = int((time.time() * 1000) % 2**31)
    return new_seed


def set_all_seeds_from_config(config) -> int:
    """
    Convenience function: read seed from config dict/yaml.
    """
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", True)
    set_seed(seed, deterministic)
    return seed
