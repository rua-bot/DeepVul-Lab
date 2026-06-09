"""Dataset loaders that normalize Devign and DiverseVul to one record schema.

Each loader returns plain Python dicts with these fields:

    id:        Stable string identifier, prefixed by dataset name.
    func:      Raw function source code (str).
    label:     Binary vulnerability label (int, 1 = vulnerable).
    project:   Originating project / repository name (str).
    commit_id: Fixing-commit SHA when available, else "" (str).
    cwe:       List of CWE strings, e.g. ["CWE-787"] (empty if unknown).
    dataset:   Source dataset name ("devign" or "diversevul").

Both datasets are expected to be present in the local HuggingFace cache; callers
that must guarantee no network access should set HF_HUB_OFFLINE=1 beforehand.
"""

from __future__ import annotations

from datasets import load_dataset

DEVIGN_REPO = "DetectVul/devign"
DIVERSEVUL_REPO = "claudios/DiverseVul"


def load_devign() -> dict[str, list[dict]]:
    """Load Devign, preserving its predefined train/validation/test splits.

    Devign covers only two projects (FFmpeg, QEMU), so a project-level split is
    not meaningful; the dataset's own split assignment is kept here.

    Returns:
        Mapping from split name ("train", "validation", "test") to a list of
        records in the unified schema.
    """
    ds = load_dataset(DEVIGN_REPO)
    out: dict[str, list[dict]] = {}
    for split in ds.keys():
        records = []
        for i, ex in enumerate(ds[split]):
            records.append(
                {
                    "id": f"devign-{split}-{i}",
                    "func": ex["func"],
                    "label": int(bool(ex["target"])),
                    "project": ex.get("project", ""),
                    "commit_id": ex.get("commit_id", ""),
                    "cwe": [],
                    "dataset": "devign",
                }
            )
        out[split] = records
    return out


def load_diversevul() -> list[dict]:
    """Load DiverseVul as a single flat list (it ships without splits).

    Returns:
        List of records in the unified schema. The caller is responsible for
        creating project-level splits.
    """
    ds = load_dataset(DIVERSEVUL_REPO)
    split = list(ds.keys())[0]
    records = []
    for i, ex in enumerate(ds[split]):
        cwe = ex.get("cwe") or []
        records.append(
            {
                "id": f"diversevul-{i}",
                "func": ex["func"],
                "label": int(ex["target"]),
                "project": ex.get("project", ""),
                "commit_id": ex.get("commit_id", ""),
                "cwe": list(cwe),
                "dataset": "diversevul",
            }
        )
    return records
