"""Code normalization and duplicate removal.

Deduplication matters because vulnerability datasets are known to contain exact
and near-duplicate functions; if duplicates straddle train/test they inflate
reported metrics. This module provides:

  * normalize_code: strip comments and collapse whitespace.
  * exact_dedup: drop records sharing a normalized-code hash.
  * near_dedup: drop near-duplicates via MinHash + LSH over token shingles.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from datasketch import MinHash, MinHashLSH

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")


def normalize_code(code: str) -> str:
    """Return a normalized form of a C/C++ function for duplicate detection.

    Removes block and line comments and collapses all runs of whitespace to a
    single space. This is an approximation (it does not parse string literals),
    which is acceptable for hashing-based duplicate detection.

    Args:
        code: Raw function source.

    Returns:
        Normalized, single-spaced source string.
    """
    code = _BLOCK_COMMENT.sub(" ", code)
    code = _LINE_COMMENT.sub(" ", code)
    return _WHITESPACE.sub(" ", code).strip()


def _hash(text: str) -> str:
    """Return a hex SHA-1 digest of ``text``."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def exact_dedup(records: list[dict]) -> tuple[list[dict], int]:
    """Remove records whose normalized code is byte-identical to an earlier one.

    Args:
        records: Records in the unified schema (must contain "func").

    Returns:
        Tuple of (kept_records, num_removed). The first occurrence is kept.
    """
    seen: set[str] = set()
    kept = []
    for rec in records:
        h = _hash(normalize_code(rec["func"]))
        if h in seen:
            continue
        seen.add(h)
        kept.append(rec)
    return kept, len(records) - len(kept)


def _shingles(text: str, k: int = 5) -> set[bytes]:
    """Return the set of k-gram token shingles for ``text``."""
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens).encode("utf-8")} if tokens else {b""}
    return {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}


def _minhash(text: str, num_perm: int) -> MinHash:
    """Build a MinHash sketch from the shingles of ``text``."""
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        mh.update(sh)
    return mh


def near_dedup(
    records: list[dict],
    threshold: float = 0.85,
    num_perm: int = 64,
    progress_every: int = 50_000,
) -> tuple[list[dict], int]:
    """Remove near-duplicate records using MinHash LSH.

    Two records are considered duplicates when the estimated Jaccard similarity
    of their normalized-code shingle sets is at least ``threshold``. The first
    occurrence in iteration order is kept.

    Args:
        records: Records in the unified schema.
        threshold: Jaccard similarity threshold for considering a duplicate.
        num_perm: Number of MinHash permutations (accuracy/speed trade-off).
        progress_every: Print progress every this many processed records.

    Returns:
        Tuple of (kept_records, num_removed).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for idx, rec in enumerate(records):
        mh = _minhash(normalize_code(rec["func"]), num_perm)
        if lsh.query(mh):
            continue
        lsh.insert(rec["id"], mh)
        kept.append(rec)
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"    near_dedup: processed {idx + 1}/{len(records)}, kept {len(kept)}")
    return kept, len(records) - len(kept)


def normalized_hashes(records: Iterable[dict]) -> set[str]:
    """Return the set of normalized-code hashes for a collection of records.

    Useful for cross-split leakage checks.

    Args:
        records: Records in the unified schema.

    Returns:
        Set of hex SHA-1 digests of the normalized code.
    """
    return {_hash(normalize_code(rec["func"])) for rec in records}


def dedup_across_splits(
    ordered_splits: list[tuple[str, list[dict]]],
    threshold: float = 0.85,
    num_perm: int = 64,
) -> tuple[dict[str, list[dict]], int]:
    """Deduplicate records across pre-defined splits, preserving split labels.

    Records are processed in the given order while sharing one exact-hash set and
    one MinHash LSH index, so a record is dropped if it duplicates anything seen
    in an earlier split. Passing splits as [train, validation, test] therefore
    removes both within-split duplicates and cross-split leakage, always keeping
    the copy that appears in the earliest split.

    Args:
        ordered_splits: List of (split_name, records) in priority order.
        threshold: Jaccard similarity threshold for near-duplicates.
        num_perm: Number of MinHash permutations.

    Returns:
        Tuple of (mapping split_name -> kept records, total_removed).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen_exact: set[str] = set()
    out: dict[str, list[dict]] = {}
    removed = 0
    uid = 0
    for name, recs in ordered_splits:
        kept = []
        for rec in recs:
            norm = normalize_code(rec["func"])
            h = _hash(norm)
            if h in seen_exact:
                removed += 1
                continue
            mh = _minhash(norm, num_perm)
            if lsh.query(mh):
                removed += 1
                continue
            seen_exact.add(h)
            lsh.insert(f"u{uid}", mh)
            uid += 1
            kept.append(rec)
        out[name] = kept
    return out, removed


def remove_hashes(records: list[dict], blocked: set[str]) -> tuple[list[dict], int]:
    """Drop records whose normalized-code hash is in ``blocked``.

    Args:
        records: Records to filter.
        blocked: Hashes to exclude (e.g. hashes already present in train).

    Returns:
        Tuple of (kept_records, num_removed).
    """
    kept = [rec for rec in records if _hash(normalize_code(rec["func"])) not in blocked]
    return kept, len(records) - len(kept)
