"""Load the processed JSONL splits produced by build_dataset.py.

Reads code/data/processed/<name>/{train,validation,test}.jsonl into in-memory
HuggingFace Datasets, avoiding the datasets cache layer entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_processed(name: str) -> dict[str, Dataset]:
    """Load the processed splits for a dataset by name.

    Args:
        name: Dataset directory name ("devign" or "diversevul").

    Returns:
        Mapping from split name to a HuggingFace Dataset with the unified schema.

    Raises:
        FileNotFoundError: If the dataset directory or a split file is missing.
    """
    root = PROCESSED_ROOT / name
    if not root.exists():
        raise FileNotFoundError(f"Processed dataset not found: {root}")
    splits = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split] = Dataset.from_list(_read_jsonl(path))
    return splits
