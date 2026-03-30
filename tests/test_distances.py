"""Tests for pybelief.distances — distances, divergences, and similarities."""

import math
import pytest
from pybelief import MassFunction
from pybelief.distances import (
    jousselme,
    euclidean,
    bhattacharyya,
    tessem,
    conflict_distance,
    cosine_similarity,
    cosine_distance,
    deng_relative_entropy,
    inclusion_degree,
    auto_conflict,
    pignistic_l1,
    pignistic_l2,
)


FRAME = ["a", "b", "c"]


@pytest.fixture()
def m1():
    return MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})


@pytest.fixture()
def m2():
    return MassFunction(FRAME, {0b010: 0.5, 0b011: 0.2, 0b111: 0.3})


@pytest.fixture()
def m_vac():
    return MassFunction.vacuous(FRAME)


@pytest.fixture()
def m_certain_a():
    return MassFunction.certain(FRAME, "a")


@pytest.fixture()
def m_certain_b():
    return MassFunction.certain(FRAME, "b")


# ── Jousselme ─────────────────────────────────────────────────────────


class TestJousselme:
    def test_identity(self, m1):
        assert jousselme(m1, m1) == pytest.approx(0.0)

    def test_symmetry(self, m1, m2):
        assert jousselme(m1, m2) == pytest.approx(jousselme(m2, m1))

    def test_bounded(self, m1, m2):
        d = jousselme(m1, m2)
        assert 0.0 <= d <= 1.0 + 1e-9

    def test_triangle_inequality(self, m1, m2, m_vac):
        d12 = jousselme(m1, m2)
        d1v = jousselme(m1, m_vac)
        d2v = jousselme(m2, m_vac)
        assert d12 <= d1v + d2v + 1e-9

    def test_certain_vs_certain(self, m_certain_a, m_certain_b):
        d = jousselme(m_certain_a, m_certain_b)
        # D matrix: D({a},{a})=1, D({b},{b})=1, D({a},{b})=0
        # diff = m1-m2: {a}=1, {b}=-1
        # d^T D d = 1*1*1 + 1*(-1)*0 + (-1)*1*0 + (-1)*(-1)*1 = 2
        # jousselme = sqrt(0.5 * 2) = 1.0
        assert d == pytest.approx(1.0)

    def test_vacuous_vs_certain(self, m_vac, m_certain_a):
        d = jousselme(m_vac, m_certain_a)
        assert 0.0 < d < 1.0

    def test_hand_computed(self):
        """Two simple mass functions on frame {a, b}."""
        m1 = MassFunction(["a", "b"], {0b01: 0.6, 0b11: 0.4})
        m2 = MassFunction(["a", "b"], {0b10: 0.6, 0b11: 0.4})
        # diff: {a}=0.6, {b}=-0.6, {a,b}=0
        # D({a},{a})=1, D({a},{b})=0, D({b},{b})=1
        # d^T D d = 0.6*0.6*1 + 0.6*(-0.6)*0 + (-0.6)*0.6*0 + (-0.6)*(-0.6)*1
        # = 0.36 + 0 + 0 + 0.36 = 0.72
        # result = sqrt(0.5 * 0.72) = sqrt(0.36) = 0.6
        assert jousselme(m1, m2) == pytest.approx(0.6)

    def test_incompatible_raises(self):
        m1 = MassFunction(["a", "b"], {0b11: 1.0})
        m2 = MassFunction(["x", "y"], {0b11: 1.0})
        with pytest.raises(ValueError, match="incompatible"):
            jousselme(m1, m2)


# ── Euclidean ─────────────────────────────────────────────────────────


class TestEuclidean:
    def test_identity(self, m1):
        assert euclidean(m1, m1) == pytest.approx(0.0)

    def test_symmetry(self, m1, m2):
        assert euclidean(m1, m2) == pytest.approx(euclidean(m2, m1))

    def test_hand_computed(self):
        m1 = MassFunction(["a", "b"], {0b01: 0.8, 0b11: 0.2})
        m2 = MassFunction(["a", "b"], {0b01: 0.5, 0b11: 0.5})
        # diff: {a}=0.3, {a,b}=-0.3
        # d = sqrt(0.09 + 0.09) = sqrt(0.18)
        assert euclidean(m1, m2) == pytest.approx(math.sqrt(0.18))

    def test_triangle_inequality(self, m1, m2, m_vac):
        d12 = euclidean(m1, m2)
        d1v = euclidean(m1, m_vac)
        d2v = euclidean(m2, m_vac)
        assert d12 <= d1v + d2v + 1e-9


# ── Bhattacharyya ─────────────────────────────────────────────────────


class TestBhattacharyya:
    def test_identity(self, m1):
        assert bhattacharyya(m1, m1) == pytest.approx(0.0)

    def test_symmetry(self, m1, m2):
        assert bhattacharyya(m1, m2) == pytest.approx(bhattacharyya(m2, m1))

    def test_non_negative(self, m1, m2):
        assert bhattacharyya(m1, m2) >= -1e-12

    def test_disjoint_support_is_inf(self, m_certain_a, m_certain_b):
        assert bhattacharyya(m_certain_a, m_certain_b) == float("inf")

    def test_bayesian(self):
        m1 = MassFunction.from_bayesian(["a", "b"], {"a": 0.7, "b": 0.3})
        m2 = MassFunction.from_bayesian(["a", "b"], {"a": 0.4, "b": 0.6})
        # BC = sqrt(0.7*0.4) + sqrt(0.3*0.6)
        bc = math.sqrt(0.28) + math.sqrt(0.18)
        assert bhattacharyya(m1, m2) == pytest.approx(-math.log(bc))


# ── Tessem ────────────────────────────────────────────────────────────


class TestTessem:
    def test_identity(self, m1):
        assert tessem(m1, m1) == pytest.approx(0.0)

    def test_symmetry(self, m1, m2):
        assert tessem(m1, m2) == pytest.approx(tessem(m2, m1))

    def test_bounded(self, m1, m2):
        assert 0.0 <= tessem(m1, m2) <= 1.0 + 1e-9

    def test_certain_vs_certain(self, m_certain_a, m_certain_b):
        # BetP1 = (1, 0, 0), BetP2 = (0, 1, 0), max diff = 1
        assert tessem(m_certain_a, m_certain_b) == pytest.approx(1.0)


# ── Conflict-based distance ──────────────────────────────────────────


class TestConflictDistance:
    def test_no_conflict(self, m1):
        """Self-combination may still have internal conflict."""
        d = conflict_distance(m1, m1)
        assert d >= 0.0

    def test_total_conflict_is_inf(self, m_certain_a, m_certain_b):
        assert conflict_distance(m_certain_a, m_certain_b) == float("inf")

    def test_symmetry(self, m1, m2):
        assert conflict_distance(m1, m2) == pytest.approx(
            conflict_distance(m2, m1)
        )

    def test_hand_computed(self):
        m1 = MassFunction(["a", "b"], {0b01: 0.5, 0b11: 0.5})
        m2 = MassFunction(["a", "b"], {0b10: 0.5, 0b11: 0.5})
        # K = 0.5 * 0.5 = 0.25
        # d = -log2(0.75)
        assert conflict_distance(m1, m2) == pytest.approx(-math.log2(0.75))


# ── Cosine ────────────────────────────────────────────────────────────


class TestCosine:
    def test_identity_similarity(self, m1):
        assert cosine_similarity(m1, m1) == pytest.approx(1.0)

    def test_identity_distance(self, m1):
        assert cosine_distance(m1, m1) == pytest.approx(0.0)

    def test_symmetry(self, m1, m2):
        assert cosine_similarity(m1, m2) == pytest.approx(
            cosine_similarity(m2, m1)
        )

    def test_bounded(self, m1, m2):
        s = cosine_similarity(m1, m2)
        assert -1e-9 <= s <= 1.0 + 1e-9

    def test_orthogonal(self, m_certain_a, m_certain_b):
        # {a}=1 and {b}=1 share no focal elements → dot=0
        assert cosine_similarity(m_certain_a, m_certain_b) == pytest.approx(0.0)
        assert cosine_distance(m_certain_a, m_certain_b) == pytest.approx(1.0)

    def test_distance_plus_similarity_is_one(self, m1, m2):
        s = cosine_similarity(m1, m2)
        d = cosine_distance(m1, m2)
        assert s + d == pytest.approx(1.0)


# ── Deng relative entropy ────────────────────────────────────────────


class TestDengRelativeEntropy:
    def test_identity(self, m1):
        assert deng_relative_entropy(m1, m1) == pytest.approx(0.0)

    def test_not_symmetric(self, m1, m2):
        d12 = deng_relative_entropy(m1, m2)
        d21 = deng_relative_entropy(m2, m1)
        # They should generally differ
        # At least both non-negative
        assert d12 >= -1e-9
        assert d21 >= -1e-9

    def test_inf_when_support_missing(self, m_certain_a, m_certain_b):
        # m1 has mass on {a}, m2 has mass on {b} → m2({a})=0
        assert deng_relative_entropy(m_certain_a, m_certain_b) == float("inf")

    def test_bayesian_equals_kl(self):
        """For Bayesian masses, Deng relative entropy = KL divergence."""
        m1 = MassFunction.from_bayesian(["a", "b"], {"a": 0.7, "b": 0.3})
        m2 = MassFunction.from_bayesian(["a", "b"], {"a": 0.4, "b": 0.6})
        # KL(m1||m2) = 0.7*ln(0.7/0.4) + 0.3*ln(0.3/0.6)
        kl = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)
        assert deng_relative_entropy(m1, m2) == pytest.approx(kl)


# ── Inclusion degree ─────────────────────────────────────────────────


class TestInclusionDegree:
    def test_self_inclusion(self, m1):
        """Inclusion of m in itself is the Jaccard self-similarity."""
        inc = inclusion_degree(m1, m1)
        assert 0.0 <= inc <= 1.0 + 1e-9

    def test_certain_same(self, m_certain_a):
        # m={a}=1 included in itself: |{a}∩{a}|/|{a}| * 1*1 = 1
        assert inclusion_degree(m_certain_a, m_certain_a) == pytest.approx(1.0)

    def test_certain_disjoint(self, m_certain_a, m_certain_b):
        # m1={a}=1, m2={b}=1: |{a}∩{b}|/|{a}| = 0
        assert inclusion_degree(m_certain_a, m_certain_b) == pytest.approx(0.0)

    def test_not_symmetric(self, m1, m2):
        inc12 = inclusion_degree(m1, m2)
        inc21 = inclusion_degree(m2, m1)
        # Both valid, but not necessarily equal
        assert 0.0 <= inc12 <= 1.0 + 1e-9
        assert 0.0 <= inc21 <= 1.0 + 1e-9

    def test_vacuous_m2(self, m1, m_vac):
        # m2 = {Omega}=1. For any A: |A∩Omega|/|A| = 1
        # So Inc(m1, vacuous) = sum m1(A)*1*1 = 1
        assert inclusion_degree(m1, m_vac) == pytest.approx(1.0)


# ── Auto-conflict ────────────────────────────────────────────────────


class TestAutoConflict:
    def test_bayesian(self):
        """Bayesian singletons are disjoint, so auto-conflict > 0."""
        m = MassFunction.from_bayesian(FRAME, {"a": 0.5, "b": 0.3, "c": 0.2})
        # K = 2*(0.5*0.3 + 0.5*0.2 + 0.3*0.2) = 0.62
        assert auto_conflict(m) == pytest.approx(0.62)

    def test_certain_zero(self):
        """A single focal element has zero auto-conflict."""
        m = MassFunction.certain(FRAME, "a")
        assert auto_conflict(m) == pytest.approx(0.0)

    def test_vacuous_zero(self, m_vac):
        assert auto_conflict(m_vac) == pytest.approx(0.0)

    def test_non_negative(self, m1):
        assert auto_conflict(m1) >= -1e-12

    def test_conflicting_evidence(self):
        # Mass split between disjoint singletons → high auto-conflict
        m = MassFunction(["a", "b"], {0b01: 0.5, 0b10: 0.5})
        # K = 0.5*0.5 + 0.5*0.5 = 0.5
        assert auto_conflict(m) == pytest.approx(0.5)


# ── Pignistic L1 / L2 ────────────────────────────────────────────────


class TestPignisticDistances:
    def test_l1_identity(self, m1):
        assert pignistic_l1(m1, m1) == pytest.approx(0.0)

    def test_l2_identity(self, m1):
        assert pignistic_l2(m1, m1) == pytest.approx(0.0)

    def test_l1_symmetry(self, m1, m2):
        assert pignistic_l1(m1, m2) == pytest.approx(pignistic_l1(m2, m1))

    def test_l2_symmetry(self, m1, m2):
        assert pignistic_l2(m1, m2) == pytest.approx(pignistic_l2(m2, m1))

    def test_tessem_equals_linf(self, m1, m2):
        """Tessem is exactly L-infinity on pignistic."""
        bp1 = m1.pignistic()
        bp2 = m2.pignistic()
        linf = max(abs(bp1[x] - bp2[x]) for x in FRAME)
        assert tessem(m1, m2) == pytest.approx(linf)

    def test_l2_leq_l1(self, m1, m2):
        """L2 ≤ L1 always holds."""
        assert pignistic_l2(m1, m2) <= pignistic_l1(m1, m2) + 1e-9

    def test_certain_vs_certain(self, m_certain_a, m_certain_b):
        # BetP1 = (1,0,0), BetP2 = (0,1,0)
        # L1 = |1|+|1| = 2
        assert pignistic_l1(m_certain_a, m_certain_b) == pytest.approx(2.0)
        # L2 = sqrt(1+1) = sqrt(2)
        assert pignistic_l2(m_certain_a, m_certain_b) == pytest.approx(
            math.sqrt(2)
        )


# ── Cross-measure consistency ─────────────────────────────────────────


class TestCrossMeasure:
    def test_all_zero_on_identity(self, m1):
        """All distances should be zero when comparing m to itself."""
        assert jousselme(m1, m1) == pytest.approx(0.0)
        assert euclidean(m1, m1) == pytest.approx(0.0)
        assert bhattacharyya(m1, m1) == pytest.approx(0.0)
        assert tessem(m1, m1) == pytest.approx(0.0)
        assert cosine_distance(m1, m1) == pytest.approx(0.0)
        assert deng_relative_entropy(m1, m1) == pytest.approx(0.0)
        assert pignistic_l1(m1, m1) == pytest.approx(0.0)
        assert pignistic_l2(m1, m1) == pytest.approx(0.0)

    def test_all_positive_on_different(self, m1, m2):
        """All distances should be > 0 for different mass functions."""
        assert jousselme(m1, m2) > 1e-6
        assert euclidean(m1, m2) > 1e-6
        assert bhattacharyya(m1, m2) > 1e-6
        assert tessem(m1, m2) > 1e-6
        assert cosine_distance(m1, m2) > 1e-6
        assert pignistic_l1(m1, m2) > 1e-6
        assert pignistic_l2(m1, m2) > 1e-6


# ── TBM edge case: all mass on empty set ──────────────────────────────


class TestTBMEmptySetMass:
    """Pignistic-based distances must raise on degenerate TBM masses."""

    @pytest.fixture()
    def m_empty(self):
        """Mass function with all mass on the empty set (total conflict)."""
        # Reachable via: MassFunction.certain(f,"a") | MassFunction.certain(f,"b")
        return MassFunction(FRAME, {0: 1.0})

    @pytest.fixture()
    def m_normal(self):
        return MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})

    def test_bhattacharyya_raises(self, m_empty, m_normal):
        with pytest.raises(ValueError, match="empty set"):
            bhattacharyya(m_empty, m_normal)
        with pytest.raises(ValueError, match="empty set"):
            bhattacharyya(m_normal, m_empty)

    def test_tessem_raises(self, m_empty, m_normal):
        with pytest.raises(ValueError, match="empty set"):
            tessem(m_empty, m_normal)

    def test_pignistic_l1_raises(self, m_empty, m_normal):
        with pytest.raises(ValueError, match="empty set"):
            pignistic_l1(m_empty, m_normal)

    def test_pignistic_l2_raises(self, m_empty, m_normal):
        with pytest.raises(ValueError, match="empty set"):
            pignistic_l2(m_empty, m_normal)

    def test_non_pignistic_measures_still_work(self, m_empty, m_normal):
        """Mass-vector distances should work fine with empty-set mass."""
        assert jousselme(m_empty, m_normal) >= 0.0
        assert euclidean(m_empty, m_normal) >= 0.0
        assert cosine_distance(m_empty, m_normal) >= 0.0

    def test_conjunctive_total_conflict_produces_empty(self):
        """Verify the TBM state is actually reachable."""
        m_a = MassFunction.certain(FRAME, "a")
        m_b = MassFunction.certain(FRAME, "b")
        conj = m_a.combine_conjunctive(m_b)
        assert conj[0] == pytest.approx(1.0)
        with pytest.raises(ValueError):
            tessem(conj, m_a)
