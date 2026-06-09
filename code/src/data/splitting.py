"""Project-level train/validation/test splitting.

Splitting by project (rather than randomly over functions) prevents
near-identical functions from the same repository leaking across splits, which
is a known cause of over-optimistic vulnerability-detection metrics.
"""

from __future__ import annotations

import random
from collections import defaultdict


def project_level_split(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Partition records into splits so that no project spans two splits.

    Projects are shuffled, then greedily assigned to whichever split is furthest
    below its target share of total functions. This keeps split sizes close to
    ``ratios`` while guaranteeing project disjointness.

    Args:
        records: Records in the unified schema (must contain "project").
        ratios: Target (train, validation, test) fractions; must sum to ~1.0.
        seed: RNG seed for reproducible project shuffling.

    Returns:
        Mapping from split name to list of records.
    """
    by_project: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_project[rec["project"] or "__unknown__"].append(rec)

    total = len(records)
    targets = {
        "train": ratios[0] * total,
        "validation": ratios[1] * total,
        "test": ratios[2] * total,
    }
    assigned: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    counts = {"train": 0, "validation": 0, "test": 0}

    # Largest projects first so big chunks land before fine-tuning the balance.
    projects = sorted(by_project.items(), key=lambda kv: len(kv[1]), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(projects)
    projects.sort(key=lambda kv: len(kv[1]), reverse=True)

    for _, recs in projects:
        deficits = {s: targets[s] - counts[s] for s in assigned}
        chosen = max(deficits, key=deficits.get)
        assigned[chosen].extend(recs)
        counts[chosen] += len(recs)

    return assigned


def split_label_stats(splits: dict[str, list[dict]]) -> dict[str, dict]:
    """Summarize size, positive count and positive rate per split.

    Args:
        splits: Mapping from split name to records.

    Returns:
        Mapping from split name to {"n", "pos", "pos_rate", "n_projects"}.
    """
    stats: dict[str, dict] = {}
    for name, recs in splits.items():
        pos = sum(r["label"] for r in recs)
        n = len(recs)
        stats[name] = {
            "n": n,
            "pos": pos,
            "pos_rate": round(pos / n, 4) if n else 0.0,
            "n_projects": len({r["project"] for r in recs}),
        }
    return stats
