"""Lightweight key-context extraction for C/C++ functions.

Many functions in Devign and DiverseVul exceed the 512-token limit of the
backbone models (~48% of Devign, ~28% of DiverseVul). Naive head truncation
keeps only the signature and the top of the body, which is frequently *not*
where the vulnerability lives. This module extracts a smaller, vulnerability-
focused view of the function so that the parts most likely to carry the bug
survive truncation.

The extraction is a cheap, dependency-free static heuristic (no external parser
such as Joern or tree-sitter is required):

  1. Seed lines: lines calling memory/string/allocation/format APIs or indexing
     an array are marked as "seeds" - the operations where memory-safety bugs
     typically occur. Bare member access (``->``) is intentionally not a seed
     because it is ubiquitous in struct-heavy C and would keep nearly everything.
  2. Local window: a small symmetric line window around each seed is kept for
     immediate context.
  3. Definition proxy: the first line mentioning each seed variable is kept (an
     approximation of its declaration / first assignment) so the model sees the
     type and origin of the buffers and sizes involved.
  4. Structural context: the enclosing block headers (function signature and any
     containing ``if`` / ``for`` / ``while`` / ``switch`` headers) of every kept
     line are kept, so the slice stays syntactically readable.

Dropped runs of lines are collapsed into a single ``// ...`` gap marker, and the
surviving lines are emitted in their original order. If a function has no seed
lines, the original code is returned unchanged so the heuristic never does worse
than the full-function baseline on such inputs.

``add_structure_markers`` optionally tags seed lines with explicit sink markers
so the model can attend to the vulnerability-relevant operations directly.
"""

from __future__ import annotations

import re

# Whole-word risky APIs commonly implicated in memory-safety / injection bugs.
SINK_APIS = frozenset(
    {
        # memory / buffer
        "memcpy", "memmove", "memset", "bcopy", "bzero",
        "strcpy", "strncpy", "strcat", "strncat", "strlcpy", "strlcat",
        "sprintf", "snprintf", "vsprintf", "vsnprintf", "scanf", "sscanf",
        "fscanf", "vscanf", "gets", "fgets", "getenv", "strdup",
        # allocation / free
        "malloc", "calloc", "realloc", "free", "alloca", "kmalloc", "kzalloc",
        "kcalloc", "krealloc", "kfree", "vmalloc", "vfree", "new", "delete",
        # length / format
        "strlen", "strnlen", "wcslen", "printf", "fprintf", "vprintf",
        "system", "popen", "exec", "execl", "execlp", "execv", "execvp",
        # integer / size
        "sizeof", "memcmp", "strcmp", "strncmp",
    }
)

# Cheap regex anchors for risky constructs that are not function calls.
_INDEX_RE = re.compile(r"\w\s*\[")          # array subscript: buf[
_DEREF_RE = re.compile(r"(->|\*\s*\w|\w\s*\*)")  # pointer deref / arithmetic
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# C/C++ keywords excluded from data-dependency identifier extraction.
_C_KEYWORDS = frozenset(
    {
        "if", "else", "for", "while", "do", "switch", "case", "default",
        "break", "continue", "return", "goto", "sizeof", "typedef", "struct",
        "union", "enum", "const", "static", "extern", "register", "volatile",
        "void", "char", "short", "int", "long", "float", "double", "signed",
        "unsigned", "bool", "true", "false", "NULL", "nullptr", "auto", "inline",
        "class", "public", "private", "protected", "virtual", "template",
        "typename", "namespace", "using", "new", "delete", "this", "operator",
    }
)

# Control-flow keywords whose headers form structural context.
_CONTROL_RE = re.compile(r"\b(if|else|for|while|do|switch|case)\b")

GAP_MARKER = "    // ..."
SINK_OPEN = "/*<sink>*/"
SINK_CLOSE = "/*</sink>*/"


def _identifiers(line: str) -> set[str]:
    """Return non-keyword identifiers appearing on a line.

    Args:
        line: A single source line.

    Returns:
        Set of identifier strings, excluding C/C++ keywords.
    """
    return {t for t in _IDENT_RE.findall(line) if t not in _C_KEYWORDS}


def _is_seed(line: str) -> bool:
    """Decide whether a line touches a vulnerability-relevant operation.

    Args:
        line: A single source line.

    Returns:
        True if the line calls a risky API or indexes an array. Bare member
        access (``->``) is deliberately excluded as too common to discriminate.
    """
    tokens = set(_IDENT_RE.findall(line))
    if tokens & SINK_APIS:
        return True
    if _INDEX_RE.search(line):
        return True
    return False


def _enclosing_headers(lines: list[str]) -> list[list[int]]:
    """Compute, for each line, the indices of its enclosing block headers.

    Block headers are the lines that opened the braces currently surrounding a
    line (the function signature plus enclosing control structures). Brace
    matching is approximate but robust to most real code.

    Args:
        lines: The function's source lines.

    Returns:
        List parallel to ``lines``; entry ``i`` holds the line indices of the
        blocks enclosing line ``i``.
    """
    enclosing: list[list[int]] = []
    stack: list[int] = []
    for i, line in enumerate(lines):
        enclosing.append(list(stack))
        for ch in line:
            if ch == "{":
                stack.append(i)
            elif ch == "}":
                if stack:
                    stack.pop()
    return enclosing


def extract_key_context(code: str, window: int = 1) -> str:
    """Extract a vulnerability-focused slice of a C/C++ function.

    Args:
        code: Raw function source.
        window: Number of neighboring lines to keep on each side of a seed line.

    Returns:
        The sliced source with dropped runs collapsed into a gap marker, or the
        original code unchanged if no seed lines are found or it is short.
    """
    lines = code.split("\n")
    if len(lines) <= 1:
        return code

    seeds = [i for i, ln in enumerate(lines) if _is_seed(ln)]
    if not seeds:
        return code

    seed_vars: set[str] = set()
    for i in seeds:
        seed_vars |= _identifiers(lines[i])

    enclosing = _enclosing_headers(lines)
    keep: set[int] = set()

    for i in seeds:
        for j in range(max(0, i - window), min(len(lines), i + window + 1)):
            keep.add(j)

    # Definition proxy: keep only the first line each seed variable appears on,
    # rather than every line that uses it (which would keep almost the whole
    # function for the dataset's most common objects).
    remaining = set(seed_vars)
    for i, ln in enumerate(lines):
        if not remaining:
            break
        hit = _identifiers(ln) & remaining
        if hit:
            keep.add(i)
            remaining -= hit

    # Keep the function signature region (everything up to the first '{').
    for i, ln in enumerate(lines):
        keep.add(i)
        if "{" in ln:
            break

    # Add enclosing structural headers for every kept line.
    for i in list(keep):
        keep.update(enclosing[i])

    out: list[str] = []
    prev_kept = -1
    for i in sorted(keep):
        if i != prev_kept + 1 and out:
            out.append(GAP_MARKER)
        out.append(lines[i])
        prev_kept = i

    sliced = "\n".join(out)
    return sliced if sliced.strip() else code


def add_structure_markers(code: str) -> str:
    """Wrap seed lines with explicit sink markers around the whole function.

    Unlike :func:`extract_key_context`, this keeps every line but annotates the
    vulnerability-relevant ones, letting the model attend to them directly. The
    markers are comment-style tokens that survive BPE tokenization.

    Args:
        code: Raw function source.

    Returns:
        The annotated source; unchanged if no seed lines are found.
    """
    lines = code.split("\n")
    seeds = {i for i, ln in enumerate(lines) if _is_seed(ln)}
    if not seeds:
        return code
    out = []
    for i, ln in enumerate(lines):
        if i in seeds:
            stripped = ln.rstrip("\n")
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}{SINK_OPEN} {stripped.strip()} {SINK_CLOSE}")
        else:
            out.append(ln)
    return "\n".join(out)
