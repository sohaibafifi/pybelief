r"""Distances, divergences, and similarity measures between mass functions.

All functions take two :class:`~pybelief.mass.MassFunction` objects defined
on the same frame and return a non-negative ``float``.

Pignistic-based measures (``bhattacharyya``, ``tessem``, ``pignistic_l1``,
``pignistic_l2``) require the pignistic transform to be defined and will
raise ``ValueError`` on degenerate TBM masses where all mass sits on the
empty set.

References
----------
.. [Jousselme2001] Jousselme, A.-L., Grenier, D., & Bossé, É. (2001).
   A new distance between two bodies of evidence.
   *Information Fusion*, 2(2), 91-101.

.. [Tessem1993] Tessem, B. (1993).
   Approximations for efficient computation in the theory of evidence.
   *Artificial Intelligence*, 61(2), 315-329.

.. [Bhattacharyya1943] Bhattacharyya, A. (1943).
   On a measure of divergence between two statistical populations defined
   by their probability distributions.
   *Bulletin of the Calcutta Mathematical Society*, 35, 99-109.

.. [Deng2016] Deng, Y. (2016).
   Deng entropy.
   *Chaos, Solitons & Fractals*, 91, 549-553.

.. [Smets1990] Smets, P. (1990).
   The combination of evidence in the transferable belief model.
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
   12(5), 447-458.

.. [Zouhal1998] Zouhal, L. M., & Denœux, T. (1998).
   An evidence-theoretic k-NN rule with parameter optimization.
   *IEEE Transactions on Systems, Man, and Cybernetics - Part C*,
   28(2), 263-271.

.. [Ristic2008] Ristić, B., & Smets, P. (2008).
   Global cost of a decision in the TBM.
   *International Journal of Approximate Reasoning*, 48(2), 327-341.

.. [Daniel2006] Daniel, M. (2006).
   Conflicts within and between belief functions.
   In *Information Processing and Management of Uncertainty in
   Knowledge-Based Systems (IPMU)*, pp. 696-703.

.. [Jousselme2012] Jousselme, A.-L., & Maupin, P. (2012).
   Distances in evidence theory: Comprehensive survey and generalizations.
   *International Journal of Approximate Reasoning*, 53(2), 118-145.
"""

from __future__ import annotations

import math

from pybelief.mass import MassFunction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_compatible(m1: MassFunction, m2: MassFunction) -> None:
    if m1.frame != m2.frame:
        raise ValueError(
            f"incompatible frames: {list(m1.frame)} vs {list(m2.frame)}"
        )


def _all_keys(m1: MassFunction, m2: MassFunction) -> set[int]:
    """Union of focal-element bitmasks from both mass functions."""
    return set(m1._m) | set(m2._m)




# ---------------------------------------------------------------------------
# Jousselme distance
# ---------------------------------------------------------------------------

def jousselme(m1: MassFunction, m2: MassFunction) -> float:
    r"""Jousselme distance between two mass functions.

    .. math::
        d_J(m_1, m_2)
        = \sqrt{\tfrac{1}{2}\,(\mathbf{m_1} - \mathbf{m_2})^\top
          \mathbf{D}\,(\mathbf{m_1} - \mathbf{m_2})}

    where the Jaccard matrix entry is

    .. math::
        D(A, B) = \frac{|A \cap B|}{|A \cup B|}

    for non-empty sets, and :math:`D(\emptyset, \emptyset) = 1`,
    :math:`D(\emptyset, B) = 0`.

    This is the standard distance in the Dempster-Shafer community,
    metrically proper and bounded in :math:`[0, 1]`.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Distance in :math:`[0, 1]`.

    References
    ----------
    .. [1] Jousselme, Grenier & Bossé (2001). *Information Fusion*, 2(2), 91-101.
    .. [2] Jousselme & Maupin (2012). *Int. J. Approx. Reasoning*, 53(2), 118-145.
    """
    _check_compatible(m1, m2)

    # Difference vector (sparse)
    keys = _all_keys(m1, m2)
    diff: dict[int, float] = {}
    for k in keys:
        d = m1._m.get(k, 0.0) - m2._m.get(k, 0.0)
        if d != 0.0:
            diff[k] = d

    if not diff:
        return 0.0

    # Compute d^T D d using only non-zero diff entries
    total = 0.0
    diff_items = list(diff.items())
    for a, da in diff_items:
        for b, db in diff_items:
            if a == 0 and b == 0:
                jac = 1.0
            elif a == 0 or b == 0:
                jac = 0.0
            else:
                inter = (a & b).bit_count()
                union = (a | b).bit_count()
                jac = inter / union
            total += da * db * jac

    return math.sqrt(max(0.0, 0.5 * total))


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------

def euclidean(m1: MassFunction, m2: MassFunction) -> float:
    r"""Euclidean distance on the mass vectors.

    .. math::
        d_E(m_1, m_2) = \sqrt{\sum_A \bigl(m_1(A) - m_2(A)\bigr)^2}

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Non-negative distance.

    References
    ----------
    .. [1] Jousselme & Maupin (2012). *Int. J. Approx. Reasoning*, 53(2), 118-145.
    """
    _check_compatible(m1, m2)
    keys = _all_keys(m1, m2)
    return math.sqrt(math.fsum(
        (m1._m.get(k, 0.0) - m2._m.get(k, 0.0)) ** 2 for k in keys
    ))


# ---------------------------------------------------------------------------
# Bhattacharyya distance
# ---------------------------------------------------------------------------

def bhattacharyya(m1: MassFunction, m2: MassFunction) -> float:
    r"""Bhattacharyya distance on the pignistic probabilities.

    .. math::
        d_B(m_1, m_2)
        = -\ln\!\Bigl(\sum_x \sqrt{BetP_1(x)\,BetP_2(x)}\Bigr)

    When the Bhattacharyya coefficient is zero (disjoint supports),
    returns ``float('inf')``.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.  The pignistic transform must
        be defined (i.e., not all mass on the empty set).

    Returns
    -------
    float
        Non-negative distance (possibly ``inf``).

    Raises
    ------
    ValueError
        If either mass function has all mass on the empty set
        (pignistic transform undefined).

    References
    ----------
    .. [1] Bhattacharyya (1943). *Bull. Calcutta Math. Soc.*, 35, 99-109.
    .. [2] Zouhal & Denœux (1998). *IEEE Trans. SMC-C*, 28(2), 263-271.
    """
    _check_compatible(m1, m2)
    bp1 = m1.pignistic()
    bp2 = m2.pignistic()
    bc = math.fsum(
        math.sqrt(bp1[x] * bp2[x]) for x in m1.frame
    )
    if bc <= 0.0:
        return float("inf")
    return -math.log(min(bc, 1.0))


# ---------------------------------------------------------------------------
# Tessem distance
# ---------------------------------------------------------------------------

def tessem(m1: MassFunction, m2: MassFunction) -> float:
    r"""Tessem distance (L-infinity on pignistic probabilities).

    .. math::
        d_T(m_1, m_2) = \max_x \lvert BetP_1(x) - BetP_2(x) \rvert

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.  The pignistic transform must
        be defined (i.e., not all mass on the empty set).

    Returns
    -------
    float
        Distance in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If either mass function has all mass on the empty set.

    References
    ----------
    .. [1] Tessem (1993). *Artificial Intelligence*, 61(2), 315-329.
    """
    _check_compatible(m1, m2)
    bp1 = m1.pignistic()
    bp2 = m2.pignistic()
    return max(abs(bp1[x] - bp2[x]) for x in m1.frame)


# ---------------------------------------------------------------------------
# Conflict-based distance
# ---------------------------------------------------------------------------

def conflict_distance(m1: MassFunction, m2: MassFunction) -> float:
    r"""Conflict-based distance (weight of evidence).

    .. math::
        d_K(m_1, m_2) = -\log_2(1 - K)

    where :math:`K = \sum_{A \cap B = \emptyset} m_1(A)\,m_2(B)`.

    Returns ``float('inf')`` when :math:`K = 1` (total conflict).

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Non-negative distance (possibly ``inf``).

    References
    ----------
    .. [1] Smets (1990). *IEEE Trans. PAMI*, 12(5), 447-458.
    .. [2] Ristić & Smets (2008). *Int. J. Approx. Reasoning*, 48(2), 327-341.
    """
    _check_compatible(m1, m2)
    k = m1.conflict(m2)
    if k >= 1.0 - 1e-15:
        return float("inf")
    return -math.log2(1.0 - k)


# ---------------------------------------------------------------------------
# Cosine similarity / distance
# ---------------------------------------------------------------------------

def cosine_similarity(m1: MassFunction, m2: MassFunction) -> float:
    r"""Cosine similarity between mass vectors.

    .. math::
        \text{sim}(m_1, m_2)
        = \frac{\sum_A m_1(A)\,m_2(A)}
               {\lVert m_1 \rVert \;\lVert m_2 \rVert}

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Similarity in :math:`[0, 1]` (masses are non-negative).

    References
    ----------
    .. [1] Jousselme & Maupin (2012). *Int. J. Approx. Reasoning*, 53(2), 118-145.
    """
    _check_compatible(m1, m2)
    keys = set(m1._m) & set(m2._m)
    dot = math.fsum(m1._m[k] * m2._m[k] for k in keys)
    norm1 = math.sqrt(math.fsum(v * v for v in m1._m.values()))
    norm2 = math.sqrt(math.fsum(v * v for v in m2._m.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def cosine_distance(m1: MassFunction, m2: MassFunction) -> float:
    r"""Cosine distance: :math:`1 - \text{cosine\_similarity}`.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Distance in :math:`[0, 1]`.

    References
    ----------
    .. [1] Jousselme & Maupin (2012). *Int. J. Approx. Reasoning*, 53(2), 118-145.
    """
    return 1.0 - cosine_similarity(m1, m2)


# ---------------------------------------------------------------------------
# Deng relative entropy
# ---------------------------------------------------------------------------

def deng_relative_entropy(m1: MassFunction, m2: MassFunction) -> float:
    r"""Deng relative entropy (generalized KL divergence for mass functions).

    .. math::
        D_{\text{Deng}}(m_1 \| m_2)
        = \sum_A m_1(A)\,\ln\!\frac{m_1(A)}{m_2(A)}

    Only focal elements of *m1* contribute.  If any focal element of
    *m1* has zero mass in *m2*, returns ``float('inf')``.

    This is **not symmetric**: :math:`D(m_1 \| m_2) \neq D(m_2 \| m_1)`.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Non-negative divergence (possibly ``inf``).

    References
    ----------
    .. [1] Deng (2016). *Chaos, Solitons & Fractals*, 91, 549-553.
    """
    _check_compatible(m1, m2)
    total = 0.0
    for mask, mass1 in m1._m.items():
        if mass1 == 0.0:
            continue
        mass2 = m2._m.get(mask, 0.0)
        if mass2 == 0.0:
            return float("inf")
        total += mass1 * math.log(mass1 / mass2)
    return total


# ---------------------------------------------------------------------------
# Inclusion degree
# ---------------------------------------------------------------------------

def inclusion_degree(m1: MassFunction, m2: MassFunction) -> float:
    r"""Degree of inclusion of *m1* in *m2*.

    .. math::
        \text{Inc}(m_1, m_2)
        = \sum_A \sum_B m_1(A)\,m_2(B)\,\frac{|A \cap B|}{|A|}

    Measures how much the evidence in *m1* is "included" in *m2*.
    Returns a value in :math:`[0, 1]`.  Not symmetric.

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.

    Returns
    -------
    float
        Inclusion degree in :math:`[0, 1]`.

    References
    ----------
    .. [1] Zouhal & Denœux (1998). *IEEE Trans. SMC-C*, 28(2), 263-271.
    """
    _check_compatible(m1, m2)
    total = 0.0
    for a, ma in m1._m.items():
        if a == 0:
            continue
        card_a = a.bit_count()
        for b, mb in m2._m.items():
            inter = (a & b).bit_count()
            total += ma * mb * (inter / card_a)
    return total


# ---------------------------------------------------------------------------
# Auto-conflict
# ---------------------------------------------------------------------------

def auto_conflict(m: MassFunction) -> float:
    r"""Auto-conflict: conflict of a mass function with itself.

    .. math::
        K_{\text{auto}}(m) = \sum_{A \cap B = \emptyset} m(A)\,m(B)

    Measures the internal incoherence of the evidence.
    A categorical mass function (single focal element) always has zero
    auto-conflict.  Bayesian masses over multiple singletons have
    non-zero auto-conflict because distinct singletons are disjoint.

    Parameters
    ----------
    m : MassFunction
        A mass function.

    Returns
    -------
    float
        Auto-conflict in :math:`[0, 1]`.

    References
    ----------
    .. [1] Daniel (2006). Conflicts within and between belief functions.
       *IPMU 2006*, pp. 696-703.
    """
    return m.conflict(m)


# ---------------------------------------------------------------------------
# Pignistic L1 / L2 distances
# ---------------------------------------------------------------------------

def pignistic_l1(m1: MassFunction, m2: MassFunction) -> float:
    r"""L1 (Manhattan) distance between pignistic probabilities.

    .. math::
        d_{L1}(m_1, m_2) = \sum_x |BetP_1(x) - BetP_2(x)|

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.  The pignistic transform must
        be defined (i.e., not all mass on the empty set).

    Returns
    -------
    float
        Non-negative distance.

    Raises
    ------
    ValueError
        If either mass function has all mass on the empty set.

    References
    ----------
    .. [1] Tessem (1993). *Artificial Intelligence*, 61(2), 315-329.
    """
    _check_compatible(m1, m2)
    bp1 = m1.pignistic()
    bp2 = m2.pignistic()
    return math.fsum(abs(bp1[x] - bp2[x]) for x in m1.frame)


def pignistic_l2(m1: MassFunction, m2: MassFunction) -> float:
    r"""L2 (Euclidean) distance between pignistic probabilities.

    .. math::
        d_{L2}(m_1, m_2) = \sqrt{\sum_x (BetP_1(x) - BetP_2(x))^2}

    Parameters
    ----------
    m1, m2 : MassFunction
        Mass functions on the same frame.  The pignistic transform must
        be defined (i.e., not all mass on the empty set).

    Returns
    -------
    float
        Non-negative distance.

    Raises
    ------
    ValueError
        If either mass function has all mass on the empty set.

    References
    ----------
    .. [1] Tessem (1993). *Artificial Intelligence*, 61(2), 315-329.
    """
    _check_compatible(m1, m2)
    bp1 = m1.pignistic()
    bp2 = m2.pignistic()
    return math.sqrt(math.fsum(
        (bp1[x] - bp2[x]) ** 2 for x in m1.frame
    ))
