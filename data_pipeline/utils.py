# data_pipeline/utils.py
from pathlib import Path
import numpy as np
import random
import logging
import torch

def set_seed(seed: int = 42):
    """Set seeds for reproducibility (numpy, random, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_logger(name: str = __name__, level: int = logging.INFO):
    """Return a simple configured logger (stream handler)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

def save_npz(path: Path | str, **arrays):
    """Save multiple arrays to a compressed .npz file. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)

def load_npz(path: Path | str):
    """Load .npz with allow_pickle=True."""
    return np.load(Path(path), allow_pickle=True)

def save_torch(path: Path | str, obj):
    """Save a torch object (tensor/dict/model) to path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, p)

def load_torch(path: Path | str, map_location=None):
    """Load a torch object from path."""
    return torch.load(Path(path), map_location=map_location)
