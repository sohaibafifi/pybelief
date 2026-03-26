"""Display and I/O utilities for mass functions.

Pretty-print tables, export to CSV/JSON, and interoperability with
other belief-function toolboxes.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any

from pybelief.mass import MassFunction, _mask_to_labels


# ── Pretty-print table ────────────────────────────────────────────────

def table(
    m: MassFunction,
    *,
    columns: Sequence[str] = ("m", "Bel", "Pl", "Q"),
    sort_by: str = "cardinality",
    empty_set: bool = False,
    precision: int = 4,
) -> str:
    """Format a mass function as a text table.

    Parameters
    ----------
    m : MassFunction
        The mass function to display.
    columns : sequence of str
        Which columns to show.  Choose from
        ``"m"``, ``"Bel"``, ``"Pl"``, ``"Q"``, ``"BetP"``.
    sort_by : str
        ``"cardinality"`` (default), ``"mass"``, or ``"mask"``.
    empty_set : bool
        Whether to include the empty-set row.
    precision : int
        Decimal places for float values.

    Returns
    -------
    str
        Formatted table ready for ``print()``.
    """
    n = len(m.frame)
    frame_mask = m._frame_mask

    # Determine which subsets to show
    if empty_set:
        masks = list(range(frame_mask + 1))
    else:
        masks = list(range(1, frame_mask + 1))

    # Sort
    if sort_by == "mass":
        masks.sort(key=lambda k: m[k], reverse=True)
    elif sort_by == "mask":
        masks.sort()
    else:  # cardinality
        masks.sort(key=lambda k: (k.bit_count(), k))

    # Precompute pignistic if needed
    betp: dict[str, float] | None = None
    if "BetP" in columns:
        try:
            betp = m.pignistic()
        except ValueError:
            betp = None

    # Build column values
    def _fmt(v: float) -> str:
        return f"{v:.{precision}f}"

    def _set_str(mask: int) -> str:
        if mask == 0:
            return "{}"
        labels = sorted(_mask_to_labels(m.frame, mask))
        return "{" + ", ".join(labels) + "}"

    col_header = "Set"
    rows: list[list[str]] = []
    headers = [col_header] + list(columns)

    for mask in masks:
        row = [_set_str(mask)]
        for col in columns:
            if col == "m":
                row.append(_fmt(m[mask]))
            elif col == "Bel":
                row.append(_fmt(m.belief(mask)))
            elif col == "Pl":
                row.append(_fmt(m.plausibility(mask)))
            elif col == "Q":
                row.append(_fmt(m.commonality(mask)))
            elif col == "BetP":
                if mask.bit_count() == 1 and betp is not None:
                    idx = mask.bit_length() - 1
                    row.append(_fmt(betp[m.frame[idx]]))
                else:
                    row.append("-")
            else:
                row.append("?")
        rows.append(row)

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format
    sep = "  "
    lines: list[str] = []
    header_line = sep.join(h.rjust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append(sep.join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(cell.rjust(widths[i]) for i, cell in enumerate(row)))

    return "\n".join(lines)


# ── CSV export ─────────────────────────────────────────────────────────

def to_csv(
    m: MassFunction,
    *,
    columns: Sequence[str] = ("m", "Bel", "Pl", "Q"),
) -> str:
    """Export a mass function as a CSV string.

    Returns a string with header row and one row per non-empty subset.
    """
    n = len(m.frame)
    frame_mask = m._frame_mask

    betp: dict[str, float] | None = None
    if "BetP" in columns:
        try:
            betp = m.pignistic()
        except ValueError:
            betp = None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Set"] + list(columns))

    for mask in range(1, frame_mask + 1):
        labels = sorted(_mask_to_labels(m.frame, mask))
        set_str = "{" + ", ".join(labels) + "}"
        row: list[Any] = [set_str]
        for col in columns:
            if col == "m":
                row.append(m[mask])
            elif col == "Bel":
                row.append(m.belief(mask))
            elif col == "Pl":
                row.append(m.plausibility(mask))
            elif col == "Q":
                row.append(m.commonality(mask))
            elif col == "BetP":
                if mask.bit_count() == 1 and betp is not None:
                    idx = mask.bit_length() - 1
                    row.append(betp[m.frame[idx]])
                else:
                    row.append("")
        writer.writerow(row)

    return output.getvalue()


# ── JSON export ────────────────────────────────────────────────────────

def to_json(
    m: MassFunction,
    *,
    indent: int | None = 2,
    include_transforms: bool = False,
) -> str:
    """Export a mass function as a JSON string.

    Parameters
    ----------
    m : MassFunction
        The mass function to serialize.
    indent : int or None
        JSON indentation.
    include_transforms : bool
        If True, include belief, plausibility, and commonality for
        every subset, plus pignistic probabilities.
    """
    data: dict[str, Any] = m.to_dict()

    if include_transforms:
        bf = m.belief_function()
        pf = m.plausibility_function()
        cf = m.commonality_function()
        frame_mask = m._frame_mask

        transforms: dict[str, dict[str, float]] = {}
        for mask in range(1, frame_mask + 1):
            labels = sorted(_mask_to_labels(m.frame, mask))
            key = str(labels)
            transforms[key] = {
                "Bel": bf[mask],
                "Pl": pf[mask],
                "Q": cf[mask],
            }
        data["transforms"] = transforms

        try:
            data["pignistic"] = m.pignistic()
        except ValueError:
            pass

    return json.dumps(data, indent=indent)


# ── ibelief (R) compatibility ──────────────────────────────────────────

def to_ibelief(m: MassFunction) -> list[float]:
    """Export mass values as a vector compatible with R's ``ibelief`` package.

    Returns a list of length ``2^n`` where index *i* is the mass of
    the subset with bitmask *i*.  Index 0 corresponds to the empty set.

    In R/ibelief, subsets are typically 1-indexed from the empty set.
    This output uses 0-indexed bitmasks; to use in R::

        # R code
        library(ibelief)
        m <- c(...)  # paste the vector, drop index 0 if needed
    """
    n = len(m.frame)
    size = 1 << n
    vec = [0.0] * size
    for mask, mass in m._m.items():
        vec[mask] = mass
    return vec


def from_ibelief(
    frame: Sequence[str], vector: Sequence[float]
) -> MassFunction:
    """Import a mass vector from R's ``ibelief`` format.

    Parameters
    ----------
    frame : sequence of str
        Frame of discernment labels.
    vector : sequence of float
        Mass values of length ``2^n``, where index *i* is the mass
        of subset with bitmask *i*.
    """
    n = len(frame)
    expected = 1 << n
    if len(vector) != expected:
        raise ValueError(
            f"vector length {len(vector)} does not match "
            f"2^{n} = {expected} for frame of size {n}"
        )
    focal: dict[int, float] = {}
    for i, v in enumerate(vector):
        if v != 0.0:
            focal[i] = v
    return MassFunction(frame, focal)


# ── MATLAB BFT compatibility ──────────────────────────────────────────

def to_matlab(m: MassFunction) -> str:
    """Export as a MATLAB-compatible vector string.

    Returns a string like ``[0 0.3 0 0.2 0 0 0 0.5]`` that can
    be pasted directly into MATLAB or the Belief Functions Toolbox.
    """
    vec = to_ibelief(m)
    return "[" + " ".join(f"{v:g}" for v in vec) + "]"


# ── Credal set ─────────────────────────────────────────────────────────

def credal_set_constraints(
    m: MassFunction,
) -> tuple[list[str], list[tuple[float, float]]]:
    """Return the credal set as per-element probability bounds.

    The credal set is ``{P : Bel(A) ≤ P(A) ≤ Pl(A) for all A}``.
    For singletons this simplifies to
    ``Bel({x}) ≤ P(x) ≤ Pl({x})`` per element.

    Returns
    -------
    labels : list of str
        Frame element labels.
    bounds : list of (lower, upper)
        Probability bounds for each element.
    """
    n = len(m.frame)
    labels = list(m.frame)
    bounds = []
    for i in range(n):
        bit = 1 << i
        bounds.append((m.belief(bit), m.plausibility(bit)))
    return labels, bounds


def credal_set_vertices(m: MassFunction) -> list[dict[str, float]]:
    """Enumerate extreme points of the credal set.

    The credal set is the convex polytope
    ``{P : Bel(A) ≤ P(A) ≤ Pl(A) for all A, sum P = 1}``.

    For small frames (≤ ~8 elements), enumerates vertices via
    the permutation method: each permutation of the frame elements
    yields a vertex where probability is assigned greedily to
    satisfy the belief constraints.

    For Bayesian mass functions the credal set is a single point.

    Returns
    -------
    list of dict[str, float]
        Each dict maps frame labels to probabilities at that vertex.

    Raises
    ------
    ValueError
        If the frame is too large (> 10 elements).
    """
    n = len(m.frame)
    if n > 10:
        raise ValueError(
            f"credal set vertex enumeration is O(n!); "
            f"frame size {n} is too large (max 10)"
        )

    from itertools import permutations

    vertices: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    for perm in permutations(range(n)):
        # For this permutation, assign probabilities greedily.
        # Process elements in permutation order. Each element gets
        # at least Bel({x}). Then distribute remaining mass to
        # satisfy the growing coalition constraints.
        p = [0.0] * n
        remaining = 1.0

        # Build coalitions in permutation order
        coalition = 0
        for idx in perm:
            bit = 1 << idx
            coalition |= bit
            # The coalition so far must get at least Bel(coalition)
            bel_c = m.belief(coalition)
            assigned = sum(p[perm[j]] for j in range(perm.index(idx)))
            # This element gets: Bel(coalition) - already assigned
            p[idx] = max(0.0, bel_c - assigned)

        # Normalize: ensure sum = 1
        total = sum(p)
        if total > 0:
            # Adjust last element to ensure sum = 1
            deficit = 1.0 - total
            p[perm[-1]] += deficit

        # Round for dedup
        key = tuple(round(v, 12) for v in p)
        if key not in seen:
            seen.add(key)
            vertices.append({m.frame[i]: p[i] for i in range(n)})

    return vertices
