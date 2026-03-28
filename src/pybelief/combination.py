"""Combination rules for Dempster-Shafer mass functions.

This module implements fusion operators beyond the basic conjunctive,
Dempster, and Yager rules already in :class:`~pybelief.mass.MassFunction`.

References
----------
.. [Shafer1976] G. Shafer, "A Mathematical Theory of Evidence",
   Princeton University Press, 1976.
.. [Smets1990] P. Smets, "The combination of evidence in the Transferable
   Belief Model", IEEE Trans. PAMI, 12(5), pp. 447-458, 1990.
.. [DuboisPrade1988] D. Dubois and H. Prade, "Representation and combination
   of uncertainty with belief functions and possibility measures",
   Computational Intelligence, 4(3), pp. 244-264, 1988.
.. [SmarandacheDezert2006] F. Smarandache and J. Dezert, "Proportional
   Conflict Redistribution Rules for Information Fusion", in Advances and
   Applications of DSmT for Information Fusion, vol. 2, ch. 1, 2006.
.. [Murphy2000] C.K. Murphy, "Combining belief functions when evidence
   conflicts", Decision Support Systems, 29(1), pp. 1-9, 2000.
.. [Denoeux2008] T. Denœux, "Conjunctive and disjunctive combination of
   belief functions induced by nondistinct bodies of evidence",
   Artificial Intelligence, 172(2-3), pp. 234-264, 2008.
.. [Smets1995] P. Smets, "The canonical decomposition of a weighted belief",
   in Proc. IJCAI, pp. 1896-1901, 1995.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybelief.mass import MassFunction


# ---------------------------------------------------------------------------
# Disjunctive rule  [Smets1990]
# ---------------------------------------------------------------------------

def combine_disjunctive(m1: MassFunction, m2: MassFunction) -> MassFunction:
    r"""Disjunctive combination rule.

    .. math::
        m_{12}(C) = \sum_{A \cup B = C} m_1(A)\,m_2(B)

    Dual of the conjunctive rule - uses union instead of intersection.
    Appropriate when at least one source is reliable but we do not know
    which one.

    References: [Smets1990]_, eq. (4).

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    MassFunction
        The disjunctive combination (always normalized if inputs are).
    """
    from pybelief.mass import MassFunction as MF
    m1._check_compatible(m2)
    result: dict[int, float] = defaultdict(float)
    for a, ma in m1._m.items():
        for b, mb in m2._m.items():
            result[a | b] += ma * mb
    return MF(m1.frame, dict(result))


# ---------------------------------------------------------------------------
# Dubois-Prade rule  [DuboisPrade1988]
# ---------------------------------------------------------------------------

def combine_dubois_prade(m1: MassFunction, m2: MassFunction) -> MassFunction:
    r"""Dubois-Prade combination rule.

    Non-conflicting pairs combine via intersection (like the conjunctive
    rule).  Conflicting pairs (where :math:`A \cap B = \emptyset`)
    assign their mass to the union :math:`A \cup B`.

    .. math::
        m_{DP}(C) = \sum_{\substack{A \cap B = C \\ C \neq \emptyset}}
                     m_1(A)\,m_2(B)
                   + \sum_{\substack{A \cup B = C \\ A \cap B = \emptyset}}
                     m_1(A)\,m_2(B)

    References: [DuboisPrade1988]_, Sec. 3.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    MassFunction
        The Dubois-Prade combination.
    """
    from pybelief.mass import MassFunction as MF
    m1._check_compatible(m2)
    result: dict[int, float] = defaultdict(float)
    for a, ma in m1._m.items():
        for b, mb in m2._m.items():
            intersection = a & b
            if intersection != 0:
                # Non-conflicting: mass goes to intersection
                result[intersection] += ma * mb
            else:
                # Conflicting: mass goes to union
                result[a | b] += ma * mb
    return MF(m1.frame, dict(result))


# ---------------------------------------------------------------------------
# PCR6 (Proportional Conflict Redistribution #6)  [SmarandacheDezert2006]
# ---------------------------------------------------------------------------

def combine_pcr6(m1: MassFunction, m2: MassFunction) -> MassFunction:
    r"""Proportional Conflict Redistribution rule #6 (PCR6).

    First computes the conjunctive combination.  Then, for every
    conflicting pair :math:`(A, B)` with :math:`A \cap B = \emptyset`,
    redistributes the conflict mass :math:`m_1(A)\,m_2(B)` back to
    :math:`A` and :math:`B` in proportion to their individual masses:

    .. math::
        \text{to } A: \frac{m_1(A)^2\,m_2(B)}{m_1(A) + m_2(B)}, \quad
        \text{to } B: \frac{m_2(B)^2\,m_1(A)}{m_1(A) + m_2(B)}

    References: [SmarandacheDezert2006]_, Definition 5.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    MassFunction
        The PCR6 combination.
    """
    from pybelief.mass import MassFunction as MF
    m1._check_compatible(m2)

    # Start with the conjunctive part (non-conflicting intersections)
    result: dict[int, float] = defaultdict(float)
    for a, ma in m1._m.items():
        for b, mb in m2._m.items():
            intersection = a & b
            if intersection != 0:
                result[intersection] += ma * mb

    # Redistribute conflict proportionally
    for a, ma in m1._m.items():
        for b, mb in m2._m.items():
            if a & b != 0:
                continue  # no conflict
            denom = ma + mb
            if denom == 0:
                continue
            # Conflict mass ma*mb is split proportionally
            # Fraction going to A: ma / (ma + mb)
            # Fraction going to B: mb / (ma + mb)
            conflict = ma * mb
            result[a] += conflict * ma / denom
            result[b] += conflict * mb / denom

    return MF(m1.frame, dict(result))


# ---------------------------------------------------------------------------
# Murphy's average combination  [Murphy2000]
# ---------------------------------------------------------------------------

def combine_murphy(*masses: MassFunction) -> MassFunction:
    r"""Murphy's averaging-based combination.

    Computes the element-wise average of all input mass functions, then
    applies Dempster's rule :math:`(n-1)` times on the average with
    itself.

    This simple heuristic mitigates the effect of highly conflicting
    sources by first "smoothing" the evidence before fusing.

    References: [Murphy2000]_, Sec. 3.

    Parameters
    ----------
    *masses : MassFunction
        Two or more mass functions on the same frame.

    Returns
    -------
    MassFunction
        The Murphy combination.

    Raises
    ------
    ValueError
        If fewer than two mass functions are provided or frames differ.
    """
    from pybelief.mass import MassFunction as MF
    if len(masses) < 2:
        raise ValueError("at least two mass functions are required")
    ref = masses[0]
    for other in masses[1:]:
        ref._check_compatible(other)

    # Compute element-wise average
    n = len(masses)
    avg: dict[int, float] = defaultdict(float)
    for m in masses:
        for mask, mass in m._m.items():
            avg[mask] += mass / n

    m_avg = MF(ref.frame, dict(avg))

    # Apply Dempster's rule (n-1) times on m_avg with itself
    result = m_avg
    for _ in range(n - 1):
        result = result.combine_dempster(m_avg)
    return result


# ---------------------------------------------------------------------------
# Weight function helpers (canonical decomposition)  [Smets1995, Denoeux2008]
# ---------------------------------------------------------------------------

def _commonality_to_weight(q: list[float], n: int) -> list[float]:
    r"""Compute the weight function from the commonality function.

    Uses the Möbius inversion in the log domain on the superset lattice:

    .. math::
        \ln w(A) = \sum_{B \supseteq A} (-1)^{|B \setminus A| + 1}
                   \ln q(B)

    References: [Smets1995]_, eq. (16).

    Parameters
    ----------
    q : list of float
        Commonality values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    list of float
        Weight values indexed by bitmask (length 2^n).
        ``w[0]`` is undefined (set to 1.0).
    """
    size = 1 << n
    frame_mask = size - 1

    # ln(q) table
    ln_q = [0.0] * size
    for s in range(size):
        if q[s] > 0:
            ln_q[s] = math.log(q[s])
        else:
            # q(A)=0 means dogmatic mass function - cannot decompose
            ln_q[s] = float("-inf")

    # Superset Möbius inversion of ln(q) gives ln(w)
    # ln_w(A) = sum_{B⊇A} (-1)^{|B\A|+1} * ln_q(B)
    # This is equivalent to applying superset-sum transform with alternating
    # signs - implemented via the fast zeta/Möbius on the complement lattice.
    ln_w = [0.0] * size
    for s in range(size):
        if s == 0:
            continue
        total = 0.0
        # Iterate over supersets of s
        complement = frame_mask ^ s
        sub = complement
        while True:
            superset = s | sub
            diff_bits = (superset ^ s).bit_count()  # |B \ A|
            sign = (-1) ** (diff_bits + 1)
            total += sign * ln_q[superset]
            if sub == 0:
                break
            sub = (sub - 1) & complement
        ln_w[s] = total

    w = [1.0] * size  # w[0] = 1 by convention
    for s in range(1, size):
        if s == frame_mask:
            w[s] = 1.0  # w(Omega) is not part of the decomposition
            continue
        if ln_w[s] == float("-inf") or ln_w[s] == float("inf"):
            w[s] = 0.0
        else:
            w[s] = math.exp(ln_w[s])
    return w


def _weight_to_commonality(w: list[float], n: int) -> list[float]:
    r"""Reconstruct the commonality function from weight values.

    The canonical decomposition gives:

    .. math::
        q(B) = \prod_{\emptyset \neq A \subsetneq \Omega,\; B \not\subseteq A} w(A)

    In the log domain:
    ``ln q(B) = T - SupersetSum(B)``
    where ``T = sum of ln w for proper non-empty subsets`` and
    ``SupersetSum(B)`` is the superset-sum of ``ln w`` at ``B``.

    References: [Smets1995]_.

    Parameters
    ----------
    w : list of float
        Weight values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    list of float
        Commonality values indexed by bitmask (length 2^n).
    """
    size = 1 << n
    frame_mask = size - 1

    # ln(w) for proper non-empty subsets only; 0 elsewhere
    ln_w = [0.0] * size
    for s in range(1, frame_mask):
        if w[s] > 0:
            ln_w[s] = math.log(w[s])
        else:
            ln_w[s] = float("-inf")
    # ln_w[0] = 0 and ln_w[frame_mask] = 0 (not part of decomposition)

    # Superset-sum transform: ss[B] = sum_{A⊇B} ln_w[A]
    # Since ln_w[0]=0 and ln_w[frame_mask]=0, this equals
    # the sum over proper non-empty supersets of B.
    ss = list(ln_w)
    for i in range(n):
        bit = 1 << i
        for s in range(size):
            if not (s & bit):
                ss[s] += ss[s | bit]

    # T = total of all ln_w = ss[0] (superset-sum at empty set)
    total = ss[0]

    # q(B) = exp(T - ss[B]) for non-empty B
    q = [0.0] * size
    q[0] = 1.0  # q(∅) = total mass = 1 for normalized mass functions
    for s in range(1, size):
        val = total - ss[s]
        if math.isinf(val) and val < 0:
            q[s] = 0.0
        else:
            q[s] = math.exp(val)
    return q


def _commonality_to_mass(q: list[float], n: int) -> dict[int, float]:
    r"""Convert commonality function to mass function via Möbius inversion.

    .. math::
        m(A) = \sum_{B \supseteq A} (-1)^{|B| - |A|}\,q(B)

    This is the standard Möbius inversion on the superset lattice.

    Parameters
    ----------
    q : list of float
        Commonality values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    dict[int, float]
        ``{bitmask: mass}`` for focal elements with non-zero mass.
    """
    size = 1 << n

    # Fast Möbius inversion on superset lattice
    # Start with q, then apply the inclusion-exclusion via bit flips
    m_vals = list(q)
    for i in range(n):
        bit = 1 << i
        for s in range(size):
            if not (s & bit):
                m_vals[s] -= m_vals[s | bit]

    result: dict[int, float] = {}
    for s in range(size):
        if abs(m_vals[s]) > 1e-15:
            result[s] = m_vals[s]
    return result


# ---------------------------------------------------------------------------
# Cautious rule  [Denoeux2008]
# ---------------------------------------------------------------------------

def combine_cautious(m1: MassFunction, m2: MassFunction) -> MassFunction:
    r"""Cautious conjunctive combination rule (Denœux).

    Combines mass functions in the weight-function domain using the
    element-wise minimum.  This rule is **idempotent**:
    :math:`m \mathbin{\land_{\min}} m = m`.

    The weight function is derived from the commonality function via the
    canonical decomposition [Smets1995]_.

    Requires non-dogmatic mass functions (all commonality values > 0).

    .. math::
        w_{12}(A) = \min\bigl(w_1(A),\; w_2(A)\bigr)
        \quad \forall\, A \subsetneq \Omega,\; A \neq \emptyset

    References: [Denoeux2008]_, Sec. 4.

    Parameters
    ----------
    m1, m2 : MassFunction
        Non-dogmatic mass functions on the same frame.

    Returns
    -------
    MassFunction
        The cautious combination.

    Raises
    ------
    ValueError
        If either mass function is dogmatic (some commonality is zero).
    """
    from pybelief.mass import MassFunction as MF
    m1._check_compatible(m2)
    n = len(m1.frame)
    size = 1 << n
    frame_mask = size - 1

    # Compute full commonality functions
    q1_dict = m1.commonality_function()
    q2_dict = m2.commonality_function()
    q1 = [q1_dict[s] for s in range(size)]
    q2 = [q2_dict[s] for s in range(size)]

    # Check non-dogmatic: q(A) > 0 for all non-empty A
    for s in range(1, size):
        if q1[s] <= 0 or q2[s] <= 0:
            raise ValueError(
                "cautious rule requires non-dogmatic mass functions "
                "(all commonality values must be positive)"
            )

    # Compute weight functions
    w1 = _commonality_to_weight(q1, n)
    w2 = _commonality_to_weight(q2, n)

    # Combine: min in weight domain for proper non-empty subsets
    w12 = [1.0] * size
    for s in range(1, frame_mask):
        w12[s] = min(w1[s], w2[s])
    # w(emptyset) and w(Omega) stay at 1.0

    # Convert back: weight -> commonality -> mass
    q12 = _weight_to_commonality(w12, n)
    m12 = _commonality_to_mass(q12, n)

    return MF(m1.frame, m12)


# ---------------------------------------------------------------------------
# Bold rule  [Denoeux2008]
# ---------------------------------------------------------------------------

def _belief_to_disjunctive_weight(bel: list[float], n: int) -> list[float]:
    r"""Compute the disjunctive weight function from the belief function.

    .. math::
        \ln v(A) = \sum_{B \subseteq A} (-1)^{|A \setminus B| + 1}
                   \ln b(B)

    where :math:`b = \text{Bel}` is the belief function.

    References: [Denoeux2008]_, Sec. 3.2.

    Parameters
    ----------
    bel : list of float
        Belief values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    list of float
        Disjunctive weight values indexed by bitmask (length 2^n).
    """
    size = 1 << n
    frame_mask = size - 1

    ln_b = [0.0] * size
    for s in range(size):
        if bel[s] > 0:
            ln_b[s] = math.log(bel[s])
        else:
            ln_b[s] = float("-inf")

    # Subset Möbius inversion: ln_v(A) = sum_{∅⊂B⊆A} (-1)^{|A\B|+1} * ln_b(B)
    # Note: B=∅ is excluded because Bel(∅)=0 and ln(0)=-inf.
    ln_v = [0.0] * size
    for s in range(size):
        if s == 0:
            continue
        total = 0.0
        sub = s
        while True:
            if sub != 0:  # skip empty set
                diff_bits = (s ^ sub).bit_count()  # |A \ B|
                sign = (-1) ** (diff_bits + 1)
                total += sign * ln_b[sub]
            if sub == 0:
                break
            sub = (sub - 1) & s
        ln_v[s] = total

    v = [1.0] * size  # v[0] = 1 by convention
    for s in range(1, size):
        if s == frame_mask:
            v[s] = 1.0
            continue
        if ln_v[s] == float("-inf") or ln_v[s] == float("inf"):
            v[s] = 0.0
        else:
            v[s] = math.exp(ln_v[s])
    return v


def _disjunctive_weight_to_belief(v: list[float], n: int) -> list[float]:
    r"""Reconstruct belief function from disjunctive weight values.

    The inverse of the canonical disjunctive decomposition is:

    .. math::
        \text{Bel}(A) = \prod_{\emptyset \subset B \subseteq A} v(B)^{-1}

    In the log domain this is a negated subset-sum:
    ``ln Bel(A) = -SubsetSum(A)``.

    References: [Denoeux2008]_.

    Parameters
    ----------
    v : list of float
        Disjunctive weight values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    list of float
        Belief values indexed by bitmask (length 2^n).
    """
    size = 1 << n
    frame_mask = size - 1

    # ln(v) for proper non-empty subsets; 0 for empty set and Ω
    ln_v = [0.0] * size
    for s in range(1, frame_mask):
        if v[s] > 0:
            ln_v[s] = math.log(v[s])
        else:
            ln_v[s] = float("-inf")
    # ln_v[0] = 0 (empty set excluded), ln_v[frame_mask] = 0 (Ω excluded)

    # Subset-sum transform: ss[A] = sum_{B⊆A} ln_v[B]
    ss = list(ln_v)
    for i in range(n):
        bit = 1 << i
        for s in range(size):
            if s & bit:
                ss[s] += ss[s ^ bit]

    # Bel(A) = exp(-ss[A]) for proper non-empty A
    b = [0.0] * size
    b[0] = 0.0  # Bel(∅) = 0 always
    for s in range(1, size):
        if s == frame_mask:
            b[s] = 1.0  # Bel(Ω) = 1 always
            continue
        val = -ss[s]
        if math.isinf(val) and val < 0:
            b[s] = 0.0
        else:
            b[s] = math.exp(val)
    return b


def _belief_to_mass(bel: list[float], n: int) -> dict[int, float]:
    r"""Convert belief function to mass via Möbius inversion.

    .. math::
        m(A) = \sum_{B \subseteq A} (-1)^{|A| - |B|}\,\text{Bel}(B)

    Parameters
    ----------
    bel : list of float
        Belief values indexed by bitmask (length 2^n).
    n : int
        Number of frame elements.

    Returns
    -------
    dict[int, float]
        ``{bitmask: mass}`` for focal elements with non-zero mass.
    """
    size = 1 << n

    # Fast Möbius inversion on subset lattice
    m_vals = list(bel)
    for i in range(n):
        bit = 1 << i
        for s in range(size):
            if s & bit:
                m_vals[s] -= m_vals[s ^ bit]

    result: dict[int, float] = {}
    for s in range(size):
        if abs(m_vals[s]) > 1e-15:
            result[s] = m_vals[s]
    return result


def combine_bold(m1: MassFunction, m2: MassFunction) -> MassFunction:
    r"""Bold disjunctive combination rule (Denœux).

    Combines mass functions in the disjunctive weight-function domain
    using the element-wise maximum.  This rule is **idempotent**:
    :math:`m \mathbin{\lor_{\max}} m = m`.

    The disjunctive weight function is derived from the belief function
    via the canonical disjunctive decomposition.

    Requires non-vacuous mass functions (Bel(A) > 0 for all non-empty A
    with |A| < |Ω|, which in practice means m must assign mass to
    singletons covering all elements).

    .. math::
        v_{12}(A) = \max\bigl(v_1(A),\; v_2(A)\bigr)
        \quad \forall\, A \subsetneq \Omega,\; A \neq \emptyset

    References: [Denoeux2008]_, Sec. 5.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    MassFunction
        The bold disjunctive combination.

    Raises
    ------
    ValueError
        If either mass function has zero belief for some non-empty
        proper subset (non-vacuous condition), or if the result
        contains negative masses beyond floating-point tolerance.

    Notes
    -----
    FIXME: The bold rule can produce generalised belief functions (with
    negative masses) when the two sources differ on frames with
    three or more elements.  Masses within ``1e-10`` of zero are
    clamped; larger violations raise ``ValueError``.
    """
    from pybelief.mass import MassFunction as MF
    m1._check_compatible(m2)
    n = len(m1.frame)
    size = 1 << n
    frame_mask = size - 1

    # Compute full belief functions
    b1_dict = m1.belief_function()
    b2_dict = m2.belief_function()
    b1 = [b1_dict[s] for s in range(size)]
    b2 = [b2_dict[s] for s in range(size)]

    # Check: Bel(A) > 0 for non-empty proper subsets
    for s in range(1, frame_mask):
        if b1[s] <= 0 or b2[s] <= 0:
            raise ValueError(
                "bold rule requires non-vacuous mass functions "
                "(all belief values for proper non-empty subsets "
                "must be positive)"
            )

    # Compute disjunctive weight functions
    v1 = _belief_to_disjunctive_weight(b1, n)
    v2 = _belief_to_disjunctive_weight(b2, n)

    # Combine: max in disjunctive weight domain for proper non-empty subsets
    v12 = [1.0] * size
    for s in range(1, frame_mask):
        v12[s] = max(v1[s], v2[s])

    # Convert back: disjunctive weight -> belief -> mass
    b12 = _disjunctive_weight_to_belief(v12, n)
    m12 = _belief_to_mass(b12, n)

    # Clamp floating-point noise; reject genuine negative masses
    _EPS = 1e-10
    for mask, val in list(m12.items()):
        if val < 0:
            if val < -_EPS:
                raise ValueError(
                    "bold rule produced negative mass "
                    f"({val:.6g} on mask {mask}); the inputs likely "
                    "violate the structural conditions required for "
                    "a proper belief function result"
                )
            m12[mask] = 0.0

    return MF(m1.frame, m12)


# ---------------------------------------------------------------------------
# Multi-source fusion  [Shafer1976, Murphy2000]
# ---------------------------------------------------------------------------

def combine_multiple(
    masses: list[MassFunction],
    rule: str = "dempster",
) -> MassFunction:
    r"""Combine multiple mass functions using the specified rule.

    For sequential rules (Dempster, conjunctive, disjunctive, Yager,
    Dubois-Prade, PCR6), applies pairwise combination left to right.

    For Murphy's rule, computes the average first and then applies
    Dempster ``(n-1)`` times - which is *not* the same as sequential
    Dempster.

    Parameters
    ----------
    masses : list of MassFunction
        Two or more mass functions on the same frame.
    rule : str
        One of ``"dempster"``, ``"conjunctive"``, ``"disjunctive"``,
        ``"yager"``, ``"dubois-prade"``, ``"pcr6"``, ``"murphy"``,
        ``"cautious"``, ``"bold"``.

    Returns
    -------
    MassFunction
        The combined mass function.

    Raises
    ------
    ValueError
        If fewer than two mass functions or an unknown rule is given.
    """
    if len(masses) < 2:
        raise ValueError("at least two mass functions are required")

    rule = rule.lower().replace("_", "-").replace(" ", "-")

    if rule == "murphy":
        return combine_murphy(*masses)

    # Map rule name to pairwise function
    _pairwise = {
        "dempster": lambda a, b: a.combine_dempster(b),
        "conjunctive": lambda a, b: a.combine_conjunctive(b),
        "disjunctive": combine_disjunctive,
        "yager": lambda a, b: a.combine_yager(b),
        "dubois-prade": combine_dubois_prade,
        "pcr6": combine_pcr6,
        "cautious": combine_cautious,
        "bold": combine_bold,
    }
    if rule not in _pairwise:
        raise ValueError(
            f"unknown rule {rule!r}; choose from: "
            + ", ".join(sorted(_pairwise))
        )

    fn = _pairwise[rule]
    result = masses[0]
    for other in masses[1:]:
        result = fn(result, other)
    return result
