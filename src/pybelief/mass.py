"""Core mass function class for Dempster-Shafer belief functions.

Focal elements are stored sparsely as ``{bitmask: mass}`` dicts.
Element *i* of the frame corresponds to bit *i* of the bitmask.

Example: for frame ``("a", "b", "c")``, the set ``{a, c}`` is
represented as ``0b101 = 5``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------

def _subsets(mask: int) -> Iterator[int]:
    """Yield every subset of *mask* (including *mask* itself and 0)."""
    sub = mask
    while sub > 0:
        yield sub
        sub = (sub - 1) & mask
    yield 0


def _supersets(mask: int, frame_mask: int) -> Iterator[int]:
    """Yield every superset of *mask* within *frame_mask*."""
    complement = frame_mask ^ mask
    for sub in _subsets(complement):
        yield mask | sub


def _labels_to_mask(
    label_to_bit: dict[str, int], labels: Iterable[str]
) -> int:
    mask = 0
    for lbl in labels:
        try:
            mask |= 1 << label_to_bit[lbl]
        except KeyError:
            raise ValueError(f"label {lbl!r} is not in the frame") from None
    return mask


def _mask_to_labels(frame: tuple[str, ...], mask: int) -> frozenset[str]:
    return frozenset(frame[i] for i in range(len(frame)) if mask & (1 << i))


# ---------------------------------------------------------------------------
# MassFunction
# ---------------------------------------------------------------------------

class MassFunction:
    """A Dempster-Shafer mass function on a finite frame of discernment.

    Parameters
    ----------
    frame : sequence of str
        Element labels.  Order determines the bitmask mapping.
    focal_elements : dict[int, float], optional
        ``{bitmask: mass}`` mapping.  Mutually exclusive with
        *named_focal_elements*.
    named_focal_elements : dict[frozenset | set, float], optional
        ``{label_set: mass}`` mapping (converted to bitmasks internally).

    Examples
    --------
    >>> m = MassFunction(["a", "b", "c"], named_focal_elements={
    ...     frozenset({"a"}): 0.3,
    ...     frozenset({"a", "b"}): 0.2,
    ...     frozenset({"a", "b", "c"}): 0.5,
    ... })
    >>> m.belief({"a"})
    0.3
    """

    __slots__ = ("frame", "_m", "_frame_mask", "_label_to_bit")

    # ---- construction -----------------------------------------------------

    def __init__(
        self,
        frame: Sequence[str],
        focal_elements: dict[int, float] | None = None,
        *,
        named_focal_elements: (
            dict[frozenset[str], float] | dict[set[str], float] | None
        ) = None,
    ) -> None:
        if focal_elements is not None and named_focal_elements is not None:
            raise ValueError(
                "Provide focal_elements or named_focal_elements, not both"
            )

        self.frame: tuple[str, ...] = tuple(frame)
        if len(self.frame) != len(set(self.frame)):
            raise ValueError("frame contains duplicate labels")

        self._frame_mask: int = (1 << len(self.frame)) - 1
        self._label_to_bit: dict[str, int] = {
            lbl: i for i, lbl in enumerate(self.frame)
        }

        if named_focal_elements is not None:
            self._m: dict[int, float] = {}
            for labels, mass in named_focal_elements.items():
                mask = _labels_to_mask(self._label_to_bit, labels)
                self._m[mask] = self._m.get(mask, 0.0) + mass
        elif focal_elements is not None:
            for mask in focal_elements:
                if mask < 0 or mask > self._frame_mask:
                    raise ValueError(
                        f"bitmask {mask} is out of range for frame of "
                        f"size {len(self.frame)}"
                    )
            self._m = dict(focal_elements)
        else:
            # Default: vacuous
            self._m = {self._frame_mask: 1.0}

        for v in self._m.values():
            if v < 0:
                raise ValueError("mass values must be non-negative")

        # Remove zero entries
        self._m = {k: v for k, v in self._m.items() if v != 0.0}

    # ---- factories --------------------------------------------------------

    @classmethod
    def vacuous(cls, frame: Sequence[str]) -> MassFunction:
        """Total ignorance: all mass on the full frame."""
        n = len(frame)
        return cls(frame, {(1 << n) - 1: 1.0})

    @classmethod
    def certain(cls, frame: Sequence[str], element: str) -> MassFunction:
        """Categorical mass: all mass on a single element."""
        frame = tuple(frame)
        if element not in frame:
            raise ValueError(f"{element!r} is not in the frame")
        bit = frame.index(element)
        return cls(frame, {1 << bit: 1.0})

    @classmethod
    def from_bayesian(
        cls, frame: Sequence[str], probabilities: dict[str, float]
    ) -> MassFunction:
        """Bayesian mass function (all focal elements are singletons)."""
        frame = tuple(frame)
        label_to_bit = {lbl: i for i, lbl in enumerate(frame)}
        m: dict[int, float] = {}
        for lbl, p in probabilities.items():
            if lbl not in label_to_bit:
                raise ValueError(f"{lbl!r} is not in the frame")
            m[1 << label_to_bit[lbl]] = p
        return cls(frame, m)

    # ---- accessors --------------------------------------------------------

    def _resolve_subset(self, key: int | frozenset[str] | set[str]) -> int:
        if isinstance(key, int):
            return key
        return _labels_to_mask(self._label_to_bit, key)

    def __getitem__(self, key: int | frozenset[str] | set[str]) -> float:
        return self._m.get(self._resolve_subset(key), 0.0)

    def __contains__(self, key: int | frozenset[str] | set[str]) -> bool:  # type: ignore[override]
        return self._resolve_subset(key) in self._m

    def __len__(self) -> int:
        return len(self._m)

    def focal_elements(self) -> dict[int, float]:
        """Return a copy of ``{bitmask: mass}``."""
        return dict(self._m)

    def focal_sets(self) -> dict[frozenset[str], float]:
        """Return ``{frozenset_of_labels: mass}``."""
        return {
            _mask_to_labels(self.frame, k): v for k, v in self._m.items()
        }

    def __repr__(self) -> str:
        items = ", ".join(
            f"{_mask_to_labels(self.frame, k)}: {v}"
            for k, v in sorted(self._m.items())
        )
        return f"MassFunction(frame={list(self.frame)}, {{{items}}})"

    def __str__(self) -> str:
        lines = [f"Frame: {list(self.frame)}"]
        for mask in sorted(self._m, key=lambda k: (k.bit_count(), k)):
            labels = _mask_to_labels(self.frame, mask)
            lines.append(f"  m({set(sorted(labels))}) = {self._m[mask]:.6g}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MassFunction):
            return NotImplemented
        if set(self.frame) != set(other.frame):
            return False
        # Remap other to self's frame order if needed
        if self.frame == other.frame:
            other_m = other._m
        else:
            other_m: dict[int, float] = {}
            for mask, mass in other._m.items():
                labels = _mask_to_labels(other.frame, mask)
                new_mask = _labels_to_mask(self._label_to_bit, labels)
                other_m[new_mask] = mass
        all_keys = set(self._m) | set(other_m)
        return all(
            abs(self._m.get(k, 0.0) - other_m.get(k, 0.0)) < 1e-9
            for k in all_keys
        )

    # ---- validation -------------------------------------------------------

    def total_mass(self) -> float:
        """Sum of all masses."""
        return math.fsum(self._m.values())

    def is_valid(self) -> bool:
        """True if all masses are non-negative and sum to ~1."""
        return all(v >= 0 for v in self._m.values()) and self.is_normalized()

    def is_normalized(self, tol: float = 1e-9) -> bool:
        """True if masses sum to 1 within *tol*."""
        return abs(self.total_mass() - 1.0) < tol

    def normalize(self) -> MassFunction:
        """Return a new mass function with masses summing to 1."""
        total = self.total_mass()
        if total == 0:
            raise ValueError("cannot normalize: total mass is zero")
        return MassFunction(
            self.frame, {k: v / total for k, v in self._m.items()}
        )

    def prune(self, tol: float = 1e-12) -> MassFunction:
        """Return a new mass function without near-zero focal elements."""
        return MassFunction(
            self.frame,
            {k: v for k, v in self._m.items() if abs(v) > tol},
        )

    # ---- single-query transforms ------------------------------------------

    def belief(self, subset: int | frozenset[str] | set[str]) -> float:
        r"""Belief of a subset: :math:`Bel(A)=\sum_{B\subseteq A} m(B)`."""
        a = self._resolve_subset(subset)
        return math.fsum(m for b, m in self._m.items() if b & a == b and b != 0)

    def plausibility(self, subset: int | frozenset[str] | set[str]) -> float:
        r"""Plausibility: :math:`Pl(A)=\sum_{B \cap A \neq \emptyset} m(B)`."""
        a = self._resolve_subset(subset)
        return math.fsum(m for b, m in self._m.items() if b & a != 0)

    def commonality(self, subset: int | frozenset[str] | set[str]) -> float:
        r"""Commonality: :math:`Q(A)=\sum_{B\supseteq A} m(B)`."""
        a = self._resolve_subset(subset)
        return math.fsum(m for b, m in self._m.items() if b & a == a)

    # ---- full transforms (fast zeta) --------------------------------------

    def belief_function(self) -> dict[int, float]:
        r"""Belief for every subset via the fast zeta transform.

        Returns ``{bitmask: Bel(bitmask)}`` for all :math:`2^n` subsets.
        Complexity is :math:`O(n \cdot 2^n)`.
        """
        n = len(self.frame)
        size = 1 << n
        table = [0.0] * size
        for mask, mass in self._m.items():
            table[mask] += mass
        for i in range(n):
            bit = 1 << i
            for s in range(size):
                if s & bit:
                    table[s] += table[s ^ bit]
        # Remove mass on empty set from belief values
        empty_mass = table[0]
        return {s: table[s] - empty_mass for s in range(size)}

    def plausibility_function(self) -> dict[int, float]:
        r"""Plausibility for every subset.

        Uses the direct definition

        .. math::
            Pl(A) = \sum_{B \cap A \neq \emptyset} m(B)

        instead of ``1 - Bel(A^c)`` so the result remains correct when
        ``m(\emptyset) > 0`` (e.g. for unnormalized conjunctive masses).
        """
        n = len(self.frame)
        size = 1 << n
        table = [0.0] * size
        for mask, mass in self._m.items():
            if mask != 0:
                table[mask] += mass

        # Subset-sum transform on non-empty masses: for each A, compute the
        # total mass of non-empty subsets of A. Then
        # Pl(A) = total_nonempty_mass - mass_of_nonempty_subsets(A^c).
        nonempty_total = math.fsum(
            mass for mask, mass in self._m.items() if mask != 0
        )
        for i in range(n):
            bit = 1 << i
            for s in range(size):
                if s & bit:
                    table[s] += table[s ^ bit]

        fm = self._frame_mask
        return {s: nonempty_total - table[fm ^ s] for s in range(size)}

    def commonality_function(self) -> dict[int, float]:
        r"""Commonality for every subset via the fast superset-sum transform.

        Returns ``{bitmask: Q(bitmask)}`` for all subsets.
        """
        n = len(self.frame)
        size = 1 << n
        table = [0.0] * size
        for mask, mass in self._m.items():
            table[mask] += mass
        for i in range(n):
            bit = 1 << i
            for s in range(size):
                if not (s & bit):
                    table[s] += table[s | bit]
        return {s: table[s] for s in range(size)}

    # ---- probability transforms -------------------------------------------

    def pignistic(self) -> dict[str, float]:
        r"""Pignistic probability transform (BetP).

        .. math::
            BetP(x) = \sum_{A \ni x} \frac{m(A)}{|A|}

        Mass on the empty set is excluded and the result is normalized.
        """
        n = len(self.frame)
        probs = [0.0] * n
        # Exclude empty-set mass, renormalize
        m_empty = self._m.get(0, 0.0)
        denom = 1.0 - m_empty
        if denom == 0:
            raise ValueError("cannot compute pignistic: all mass on empty set")
        for mask, mass in self._m.items():
            if mask == 0:
                continue
            card = mask.bit_count()
            share = mass / card / denom
            for i in range(n):
                if mask & (1 << i):
                    probs[i] += share
        return {self.frame[i]: probs[i] for i in range(n)}

    def plausibility_transform(self) -> dict[str, float]:
        r"""Normalized singleton plausibilities.

        .. math::
            P_{Pl}(x) = \frac{Pl(\{x\})}{\sum_y Pl(\{y\})}
        """
        n = len(self.frame)
        pls = [self.plausibility(1 << i) for i in range(n)]
        total = math.fsum(pls)
        if total == 0:
            raise ValueError("all singleton plausibilities are zero")
        return {self.frame[i]: pls[i] / total for i in range(n)}

    # ---- operations -------------------------------------------------------

    def discount(self, alpha: float) -> MassFunction:
        r"""Shafer's discounting with reliability factor :math:`1-\alpha`.

        Parameters
        ----------
        alpha : float
            Discount rate in ``[0, 1]``.  ``0`` = no change, ``1`` = vacuous.

        .. math::
            m_\alpha(A) &= (1-\alpha)\,m(A) & A \neq \Omega \\
            m_\alpha(\Omega) &= (1-\alpha)\,m(\Omega) + \alpha
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        new_m: dict[int, float] = {}
        for mask, mass in self._m.items():
            new_m[mask] = (1 - alpha) * mass
        omega = self._frame_mask
        new_m[omega] = new_m.get(omega, 0.0) + alpha
        return MassFunction(self.frame, new_m)

    def condition(
        self, event: int | frozenset[str] | set[str]
    ) -> MassFunction:
        r"""Dempster conditioning on an event (equivalent to combining with
        a categorical mass on the event).

        Focal elements that do not intersect the event are discarded and
        the remaining masses are renormalized.
        """
        a = self._resolve_subset(event)
        if a == 0:
            raise ValueError("cannot condition on the empty set")
        new_m: dict[int, float] = defaultdict(float)
        for mask, mass in self._m.items():
            restricted = mask & a
            if restricted != 0:
                new_m[restricted] += mass
        total = math.fsum(new_m.values())
        if total == 0:
            raise ValueError("conditioning leads to total conflict")
        return MassFunction(
            self.frame, {k: v / total for k, v in new_m.items()}
        )

    # ---- combination rules ------------------------------------------------

    def _check_compatible(self, other: MassFunction) -> None:
        if self.frame != other.frame:
            raise ValueError(
                f"incompatible frames: {list(self.frame)} vs "
                f"{list(other.frame)}"
            )

    def combine_conjunctive(self, other: MassFunction) -> MassFunction:
        r"""Conjunctive combination (TBM unnormalized).

        .. math::
            m_{12}(C) = \sum_{A \cap B = C} m_1(A)\,m_2(B)

        Mass on the empty set represents conflict.
        """
        self._check_compatible(other)
        result: dict[int, float] = defaultdict(float)
        for a, ma in self._m.items():
            for b, mb in other._m.items():
                result[a & b] += ma * mb
        return MassFunction(self.frame, dict(result))

    def combine_dempster(self, other: MassFunction) -> MassFunction:
        r"""Dempster's rule (normalized conjunctive combination).

        Raises :class:`ValueError` when total conflict :math:`K = 1`.
        """
        conj = self.combine_conjunctive(other)
        k = conj._m.pop(0, 0.0)
        if abs(1.0 - k) < 1e-15:
            raise ValueError("total conflict (K=1): Dempster's rule undefined")
        return MassFunction(
            self.frame, {m: v / (1 - k) for m, v in conj._m.items()}
        )

    def combine_yager(self, other: MassFunction) -> MassFunction:
        r"""Yager's rule: conflict mass is transferred to :math:`\Omega`."""
        conj = self.combine_conjunctive(other)
        k = conj._m.pop(0, 0.0)
        omega = self._frame_mask
        m = dict(conj._m)
        m[omega] = m.get(omega, 0.0) + k
        return MassFunction(self.frame, m)

    def combine_disjunctive(self, other: MassFunction) -> MassFunction:
        r"""Disjunctive combination: :math:`m_{12}(C) = \sum_{A\cup B=C} m_1(A)\,m_2(B)`.

        Dual of the conjunctive rule.  See :func:`~pybelief.combination.combine_disjunctive`.
        """
        from pybelief.combination import combine_disjunctive
        return combine_disjunctive(self, other)

    def combine_dubois_prade(self, other: MassFunction) -> MassFunction:
        r"""Dubois-Prade rule: conflict goes to the union.

        See :func:`~pybelief.combination.combine_dubois_prade`.
        """
        from pybelief.combination import combine_dubois_prade
        return combine_dubois_prade(self, other)

    def combine_pcr6(self, other: MassFunction) -> MassFunction:
        r"""PCR6 (Proportional Conflict Redistribution #6).

        See :func:`~pybelief.combination.combine_pcr6`.
        """
        from pybelief.combination import combine_pcr6
        return combine_pcr6(self, other)

    def combine_cautious(self, other: MassFunction) -> MassFunction:
        r"""Cautious conjunctive rule (Denœux): idempotent, uses min in weight domain.

        See :func:`~pybelief.combination.combine_cautious`.
        """
        from pybelief.combination import combine_cautious
        return combine_cautious(self, other)

    def combine_bold(self, other: MassFunction) -> MassFunction:
        r"""Bold disjunctive rule (Denœux): idempotent, uses max in weight domain.

        See :func:`~pybelief.combination.combine_bold`.
        """
        from pybelief.combination import combine_bold
        return combine_bold(self, other)

    def __and__(self, other: MassFunction) -> MassFunction:
        """``m1 & m2`` - Dempster's rule."""
        return self.combine_dempster(other)

    def __or__(self, other: MassFunction) -> MassFunction:
        """``m1 | m2`` - conjunctive (TBM) combination."""
        return self.combine_conjunctive(other)

    def conflict(self, other: MassFunction) -> float:
        """Conflict mass :math:`K` between two mass functions."""
        self._check_compatible(other)
        return math.fsum(
            ma * mb
            for a, ma in self._m.items()
            for b, mb in other._m.items()
            if a & b == 0
        )

    # ---- information measures ---------------------------------------------

    def specificity(self) -> float:
        r"""Non-specificity: :math:`N(m)=\sum_A m(A)\,\log_2 |A|`."""
        return math.fsum(
            m * math.log2(mask.bit_count())
            for mask, m in self._m.items()
            if mask != 0
        )

    def entropy_deng(self) -> float:
        r"""Deng entropy (generalized Shannon entropy).

        .. math::
            H_D = -\sum_A m(A)\,\log_2 \frac{m(A)}{2^{|A|} - 1}
        """
        total = 0.0
        for mask, m in self._m.items():
            if m == 0 or mask == 0:
                continue
            card = mask.bit_count()
            total -= m * math.log2(m / (2**card - 1))
        return total

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "frame": list(self.frame),
            "focal_elements": {
                str(sorted(_mask_to_labels(self.frame, k))): v
                for k, v in self._m.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MassFunction:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        frame = tuple(data["frame"])
        named: dict[frozenset[str], float] = {}
        for key, val in data["focal_elements"].items():
            # key is a string repr of a sorted list, e.g. "['a', 'b']"
            import ast
            labels = ast.literal_eval(key)
            named[frozenset(labels)] = val
        return cls(frame, named_focal_elements=named)

    # ---- bitmask utilities (public) ---------------------------------------

    def mask_to_set(self, mask: int) -> frozenset[str]:
        """Convert a bitmask to a frozenset of labels."""
        return _mask_to_labels(self.frame, mask)

    def set_to_mask(self, labels: Iterable[str]) -> int:
        """Convert labels to a bitmask."""
        return _labels_to_mask(self._label_to_bit, labels)
