"""Dynamic GPU selection for a shared, multi-user server.

The server is shared, so we must not hard-code a device index. The policy here is
simple and robust: query `nvidia-smi`, pick the GPU with the most free memory, and
pin the process to it via CUDA_VISIBLE_DEVICES.

IMPORTANT: call `select_free_gpu()` BEFORE importing torch / creating any CUDA
context, otherwise CUDA_VISIBLE_DEVICES will not take effect for this process.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_total_mib: int
    mem_used_mib: int
    mem_free_mib: int
    util_pct: int


def query_gpus() -> list[GpuInfo]:
    """Return per-GPU info parsed from nvidia-smi (read-only)."""
    fields = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                mem_total_mib=int(parts[2]),
                mem_used_mib=int(parts[3]),
                mem_free_mib=int(parts[4]),
                util_pct=int(parts[5]),
            )
        )
    return gpus


def select_free_gpu(min_free_mib: int = 10_000, set_env: bool = True, verbose: bool = True) -> int:
    """Pick the GPU with the most free memory and pin the process to it.

    Args:
        min_free_mib: refuse to proceed if the best GPU has less free memory than
            this (guards against grabbing a nearly-full card on a busy server).
        set_env: if True, set CUDA_VISIBLE_DEVICES to the chosen physical index so
            the process sees exactly one GPU (which becomes cuda:0).
        verbose: print the decision and the full GPU table.

    Returns:
        The chosen physical GPU index.
    """
    # Explicit override for parallel launches on a shared box: when DEEPVUL_GPU
    # is set, pin to it directly and skip free-memory selection so concurrent
    # jobs do not race for the same card.
    forced = os.environ.get("DEEPVUL_GPU")
    if forced is not None and forced != "":
        idx = int(forced)
        if set_env:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        if verbose:
            print(f"DEEPVUL_GPU set: pinning to physical GPU {idx} (seen as cuda:0)")
        return idx

    gpus = query_gpus()
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPUs")

    best = max(gpus, key=lambda g: g.mem_free_mib)

    if verbose:
        print("GPU memory snapshot (MiB free / total, util%):")
        for g in gpus:
            mark = " <== selected" if g.index == best.index else ""
            print(
                f"  [{g.index}] {g.name}: {g.mem_free_mib:>7}/{g.mem_total_mib} free, "
                f"util {g.util_pct:>3}%{mark}"
            )

    if best.mem_free_mib < min_free_mib:
        raise RuntimeError(
            f"Best GPU [{best.index}] has only {best.mem_free_mib} MiB free, "
            f"below required {min_free_mib} MiB. Server is busy; try again later."
        )

    if set_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best.index)
        if verbose:
            print(f"Set CUDA_VISIBLE_DEVICES={best.index} (this process sees it as cuda:0)")

    return best.index


if __name__ == "__main__":
    select_free_gpu(min_free_mib=0)
