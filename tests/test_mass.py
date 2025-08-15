"""Comprehensive tests for pybelief.MassFunction."""

import math
import pytest
from pybelief import MassFunction


# ── helpers ────────────────────────────────────────────────────────────────

FRAME = ["a", "b", "c"]


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ── construction ──────────────────────────────────────────────────────────


class TestConstruction:
    def test_from_bitmask(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        assert len(m) == 3
        assert m[0b001] == pytest.approx(0.3)

    def test_from_named(self):
        m = MassFunction(
            FRAME,
            named_focal_elements={
                frozenset({"a"}): 0.3,
                frozenset({"a", "b"}): 0.2,
                frozenset({"a", "b", "c"}): 0.5,
            },
        )
        assert m[frozenset({"a"})] == pytest.approx(0.3)
        assert m[{"a", "b"}] == pytest.approx(0.2)

    def test_default_vacuous(self):
        m = MassFunction(FRAME)
        assert m[0b111] == pytest.approx(1.0)
        assert len(m) == 1

    def test_duplicate_frame_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            MassFunction(["a", "a", "b"])

    def test_both_args_raises(self):
        with pytest.raises(ValueError, match="not both"):
            MassFunction(FRAME, {1: 0.5}, named_focal_elements={frozenset({"a"}): 0.5})

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MassFunction(FRAME, {1: -0.5, 0b111: 1.5})

    def test_bitmask_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            MassFunction(FRAME, {0b1000: 1.0})

    def test_bad_label_raises(self):
        with pytest.raises(ValueError, match="not in the frame"):
            MassFunction(FRAME, named_focal_elements={frozenset({"x"}): 1.0})


class TestFactories:
    def test_vacuous(self):
        m = MassFunction.vacuous(FRAME)
        assert m[0b111] == pytest.approx(1.0)
        assert len(m) == 1

    def test_certain(self):
        m = MassFunction.certain(FRAME, "b")
        assert m[{"b"}] == pytest.approx(1.0)
        assert len(m) == 1

    def test_certain_bad_element(self):
        with pytest.raises(ValueError):
            MassFunction.certain(FRAME, "x")

    def test_from_bayesian(self):
        m = MassFunction.from_bayesian(FRAME, {"a": 0.5, "b": 0.3, "c": 0.2})
        assert m[{"a"}] == pytest.approx(0.5)
        assert m[{"b"}] == pytest.approx(0.3)
        assert m[{"c"}] == pytest.approx(0.2)
        assert m.is_normalized()


# ── accessors ─────────────────────────────────────────────────────────────


class TestAccessors:
    def test_getitem_missing(self):
        m = MassFunction(FRAME, {0b001: 1.0})
        assert m[0b010] == 0.0

    def test_contains(self):
        m = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        assert 0b001 in m
        assert frozenset({"a"}) in m
        assert 0b010 not in m

    def test_focal_sets(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        fs = m.focal_sets()
        assert fs[frozenset({"a"})] == pytest.approx(0.3)
        assert fs[frozenset({"a", "b", "c"})] == pytest.approx(0.7)

    def test_repr(self):
        m = MassFunction(FRAME, {0b001: 1.0})
        assert "MassFunction" in repr(m)

    def test_str(self):
        m = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        s = str(m)
        assert "Frame" in s


# ── transforms ────────────────────────────────────────────────────────────

# Reference mass function: m({a})=0.3, m({a,b})=0.2, m({a,b,c})=0.5
# Bel({a}) = 0.3
# Bel({a,b}) = 0.3 + 0.2 = 0.5
# Pl({a}) = 0.3 + 0.2 + 0.5 = 1.0
# Pl({b}) = 0.2 + 0.5 = 0.7
# Q({a}) = 0.3 + 0.2 + 0.5 = 1.0
# Q({b}) = 0.2 + 0.5 = 0.7
# Q({a,b}) = 0.2 + 0.5 = 0.7


class TestTransforms:
    @pytest.fixture()
    def m(self):
        return MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})

    def test_belief_singleton(self, m):
        assert m.belief({"a"}) == pytest.approx(0.3)

    def test_belief_pair(self, m):
        assert m.belief({"a", "b"}) == pytest.approx(0.5)

    def test_belief_full(self, m):
        assert m.belief({"a", "b", "c"}) == pytest.approx(1.0)

    def test_plausibility_a(self, m):
        assert m.plausibility({"a"}) == pytest.approx(1.0)

    def test_plausibility_b(self, m):
        assert m.plausibility({"b"}) == pytest.approx(0.7)

    def test_plausibility_c(self, m):
        assert m.plausibility({"c"}) == pytest.approx(0.5)

    def test_pl_equals_1_minus_bel_complement(self, m):
        # Pl(A) = 1 - Bel(A^c)
        fm = 0b111
        for a in range(1, fm + 1):
            pl = m.plausibility(a)
            bel_comp = m.belief(fm ^ a)
            assert pl == pytest.approx(1.0 - bel_comp), f"failed for mask {a}"

    def test_commonality_a(self, m):
        assert m.commonality({"a"}) == pytest.approx(1.0)

    def test_commonality_b(self, m):
        assert m.commonality({"b"}) == pytest.approx(0.7)

    def test_commonality_ab(self, m):
        assert m.commonality({"a", "b"}) == pytest.approx(0.7)

    def test_belief_function_matches_pointwise(self, m):
        bf = m.belief_function()
        fm = 0b111
        for s in range(fm + 1):
            assert bf[s] == pytest.approx(m.belief(s)), f"mismatch at {s}"

    def test_plausibility_function_matches_pointwise(self, m):
        pf = m.plausibility_function()
        fm = 0b111
        for s in range(fm + 1):
            assert pf[s] == pytest.approx(m.plausibility(s)), f"mismatch at {s}"

    def test_commonality_function_matches_pointwise(self, m):
        cf = m.commonality_function()
        fm = 0b111
        for s in range(fm + 1):
            assert cf[s] == pytest.approx(m.commonality(s)), f"mismatch at {s}"

    def test_bayesian_bel_equals_pl(self):
        m = MassFunction.from_bayesian(FRAME, {"a": 0.5, "b": 0.3, "c": 0.2})
        for s in range(1, 0b111 + 1):
            assert m.belief(s) == pytest.approx(m.plausibility(s))


# ── pignistic ─────────────────────────────────────────────────────────────


class TestPignistic:
    def test_reference(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        bp = m.pignistic()
        # BetP(a) = 0.3/1 + 0.2/2 + 0.5/3 = 0.5667
        assert bp["a"] == pytest.approx(0.3 + 0.1 + 0.5 / 3)
        # BetP(b) = 0.2/2 + 0.5/3 = 0.2667
        assert bp["b"] == pytest.approx(0.1 + 0.5 / 3)
        # BetP(c) = 0.5/3 = 0.1667
        assert bp["c"] == pytest.approx(0.5 / 3)
        assert sum(bp.values()) == pytest.approx(1.0)

    def test_bayesian_pignistic_equals_probs(self):
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        m = MassFunction.from_bayesian(FRAME, probs)
        bp = m.pignistic()
        for lbl, p in probs.items():
            assert bp[lbl] == pytest.approx(p)

    def test_vacuous_pignistic_is_uniform(self):
        m = MassFunction.vacuous(FRAME)
        bp = m.pignistic()
        for lbl in FRAME:
            assert bp[lbl] == pytest.approx(1 / 3)

    def test_plausibility_transform(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        pt = m.plausibility_transform()
        assert sum(pt.values()) == pytest.approx(1.0)


# ── discount / condition ──────────────────────────────────────────────────


class TestDiscountCondition:
    def test_discount_zero(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        d = m.discount(0)
        assert d == m

    def test_discount_one(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        d = m.discount(1)
        assert d == MassFunction.vacuous(FRAME)

    def test_discount_half(self):
        m = MassFunction(FRAME, {0b001: 0.6, 0b111: 0.4})
        d = m.discount(0.5)
        assert d[0b001] == pytest.approx(0.3)
        assert d[0b111] == pytest.approx(0.7)
        assert d.is_normalized()

    def test_discount_range_error(self):
        m = MassFunction.vacuous(FRAME)
        with pytest.raises(ValueError):
            m.discount(1.5)

    def test_condition(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        # Condition on {a, b} (mask 0b011)
        c = m.condition({"a", "b"})
        # m({a}) intersects {a,b} -> {a}: 0.3
        # m({a,b}) intersects {a,b} -> {a,b}: 0.2
        # m({a,b,c}) intersects {a,b} -> {a,b}: 0.5
        # total = 1.0
        assert c[{"a"}] == pytest.approx(0.3)
        assert c[{"a", "b"}] == pytest.approx(0.7)
        assert c.is_normalized()

    def test_condition_empty_raises(self):
        m = MassFunction.vacuous(FRAME)
        with pytest.raises(ValueError, match="empty set"):
            m.condition(frozenset())


# ── combination ───────────────────────────────────────────────────────────


class TestCombination:
    def test_dempster_textbook(self):
        # Classic example: two sources on frame {a, b, c}
        m1 = MassFunction(FRAME, {0b001: 0.6, 0b011: 0.3, 0b111: 0.1})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b011: 0.2, 0b111: 0.3})
        result = m1.combine_dempster(m2)
        assert result.is_normalized()

    def test_dempster_total_conflict(self):
        m1 = MassFunction(FRAME, {0b001: 1.0})
        m2 = MassFunction(FRAME, {0b010: 1.0})
        with pytest.raises(ValueError, match="total conflict"):
            m1.combine_dempster(m2)

    def test_conjunctive_conflict_on_empty(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        conj = m1.combine_conjunctive(m2)
        # conflict: m1({a})*m2({b}) = 0.25
        assert conj[0] == pytest.approx(0.25)

    def test_yager_conflict_to_omega(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        yager = m1.combine_yager(m2)
        # no mass on empty set
        assert yager[0] == 0.0
        assert yager.is_normalized()

    def test_vacuous_combination(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        v = MassFunction.vacuous(FRAME)
        assert m.combine_dempster(v) == m
        assert v.combine_dempster(m) == m

    def test_commutativity(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert m1.combine_dempster(m2) == m2.combine_dempster(m1)

    def test_associativity(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b010: 0.2, 0b111: 0.8})
        m3 = MassFunction(FRAME, {0b100: 0.1, 0b111: 0.9})
        r1 = (m1 & m2) & m3
        r2 = m1 & (m2 & m3)
        assert r1 == r2

    def test_operator_and(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert (m1 & m2) == m1.combine_dempster(m2)

    def test_operator_or(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert (m1 | m2) == m1.combine_conjunctive(m2)

    def test_incompatible_frames_raises(self):
        m1 = MassFunction(["a", "b"], {0b11: 1.0})
        m2 = MassFunction(["x", "y"], {0b11: 1.0})
        with pytest.raises(ValueError, match="incompatible"):
            m1.combine_dempster(m2)

    def test_conflict(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        assert m1.conflict(m2) == pytest.approx(0.25)


# ── information measures ──────────────────────────────────────────────────


class TestMeasures:
    def test_specificity_vacuous(self):
        m = MassFunction.vacuous(FRAME)
        # N = 1 * log2(3)
        assert m.specificity() == pytest.approx(math.log2(3))

    def test_specificity_certain(self):
        m = MassFunction.certain(FRAME, "a")
        # N = 1 * log2(1) = 0
        assert m.specificity() == pytest.approx(0.0)

    def test_entropy_deng_bayesian(self):
        # For Bayesian mass (singletons), Deng entropy = Shannon entropy
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        m = MassFunction.from_bayesian(FRAME, probs)
        shannon = -sum(p * math.log2(p) for p in probs.values())
        assert m.entropy_deng() == pytest.approx(shannon)


# ── validation / normalization ────────────────────────────────────────────


class TestValidation:
    def test_is_valid(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        assert m.is_valid()

    def test_not_normalized(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.5})
        assert not m.is_normalized()

    def test_normalize(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.5})
        n = m.normalize()
        assert n.is_normalized()
        assert n[0b001] == pytest.approx(0.3 / 0.8)

    def test_prune(self):
        m = MassFunction(FRAME, {0b001: 0.5, 0b010: 1e-15, 0b111: 0.5})
        p = m.prune()
        assert len(p) == 2


# ── equality ──────────────────────────────────────────────────────────────


class TestEquality:
    def test_equal(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        assert m1 == m2

    def test_not_equal(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        assert m1 != m2


# ── serialization ─────────────────────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self):
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        d = m.to_dict()
        m2 = MassFunction.from_dict(d)
        assert m == m2


# ── edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_element_frame(self):
        m = MassFunction(["x"], {0b1: 1.0})
        assert m.belief({"x"}) == pytest.approx(1.0)
        assert m.pignistic() == {"x": pytest.approx(1.0)}

    def test_two_element_frame(self):
        m = MassFunction(["a", "b"], {0b01: 0.4, 0b11: 0.6})
        bp = m.pignistic()
        assert bp["a"] == pytest.approx(0.4 + 0.3)
        assert bp["b"] == pytest.approx(0.3)
