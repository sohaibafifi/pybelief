"""Tests for pybelief combination rules (Phase 1)."""

import math
import pytest
from pybelief import (
    MassFunction,
    combine_disjunctive,
    combine_dubois_prade,
    combine_pcr6,
    combine_murphy,
    combine_cautious,
    combine_bold,
    combine_multiple,
)

FRAME = ["a", "b", "c"]


# ── Disjunctive rule ──────────────────────────────────────────────────────


class TestDisjunctive:
    def test_basic(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        result = m1.combine_disjunctive(m2)
        assert result.is_normalized()
        # {a}∪{b} = {a,b}, {a}∪Ω = Ω, Ω∪{b} = Ω, Ω∪Ω = Ω
        assert result[{"a", "b"}] == pytest.approx(0.25)
        assert result[{"a", "b", "c"}] == pytest.approx(0.75)

    def test_standalone_matches_method(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert combine_disjunctive(m1, m2) == m1.combine_disjunctive(m2)

    def test_commutativity(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b011: 0.2, 0b111: 0.4})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b110: 0.3, 0b111: 0.4})
        assert m1.combine_disjunctive(m2) == m2.combine_disjunctive(m1)

    def test_vacuous_absorbs(self):
        """Disjunctive combination with vacuous returns vacuous."""
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        v = MassFunction.vacuous(FRAME)
        result = m.combine_disjunctive(v)
        # Everything unioned with Omega is Omega
        assert result[0b111] == pytest.approx(1.0)

    def test_certain_elements(self):
        """Disjunctive of two certain masses gives mass on the union."""
        m1 = MassFunction.certain(FRAME, "a")
        m2 = MassFunction.certain(FRAME, "b")
        result = m1.combine_disjunctive(m2)
        assert result[{"a", "b"}] == pytest.approx(1.0)

    def test_no_empty_set_mass(self):
        """Disjunctive rule never produces mass on the empty set."""
        m1 = MassFunction(FRAME, {0b001: 0.6, 0b010: 0.4})
        m2 = MassFunction(FRAME, {0b100: 0.5, 0b010: 0.5})
        result = m1.combine_disjunctive(m2)
        assert result[0] == 0.0


# ── Dubois-Prade rule ────────────────────────────────────────────────────


class TestDuboisPrade:
    def test_no_conflict_matches_conjunctive(self):
        """When there is no conflict, DP = conjunctive."""
        m1 = MassFunction(FRAME, {0b011: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b011: 0.3, 0b111: 0.7})
        dp = m1.combine_dubois_prade(m2)
        conj = m1.combine_conjunctive(m2)
        # No conflict (all intersections non-empty), so they should match
        assert dp == conj

    def test_conflict_goes_to_union(self):
        """Conflict between {a} and {b} goes to {a,b} (not to empty set)."""
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        dp = m1.combine_dubois_prade(m2)
        assert dp.is_normalized()
        # Conflicting pair {a}∩{b}=∅ → mass goes to {a}∪{b}={a,b}
        assert dp[{"a", "b"}] == pytest.approx(0.25)
        # No mass on empty set (unlike conjunctive)
        assert dp[0] == 0.0

    def test_standalone_matches_method(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert combine_dubois_prade(m1, m2) == m1.combine_dubois_prade(m2)

    def test_commutativity(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b010: 0.2, 0b111: 0.4})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b100: 0.3, 0b111: 0.4})
        assert m1.combine_dubois_prade(m2) == m2.combine_dubois_prade(m1)

    def test_hand_computed(self):
        """Verify against hand-computed values."""
        m1 = MassFunction(["a", "b"], {0b01: 0.6, 0b10: 0.4})
        m2 = MassFunction(["a", "b"], {0b01: 0.3, 0b10: 0.7})
        dp = m1.combine_dubois_prade(m2)
        # {a}∩{a}={a}: 0.6*0.3=0.18
        # {a}∩{b}=∅ → {a}∪{b}={a,b}: 0.6*0.7=0.42
        # {b}∩{a}=∅ → {a}∪{b}={a,b}: 0.4*0.3=0.12
        # {b}∩{b}={b}: 0.4*0.7=0.28
        assert dp[{"a"}] == pytest.approx(0.18)
        assert dp[{"b"}] == pytest.approx(0.28)
        assert dp[{"a", "b"}] == pytest.approx(0.54)


# ── PCR6 rule ─────────────────────────────────────────────────────────────


class TestPCR6:
    def test_no_conflict_matches_conjunctive(self):
        """When there is no conflict, PCR6 = conjunctive."""
        m1 = MassFunction(FRAME, {0b011: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b011: 0.3, 0b111: 0.7})
        pcr6 = m1.combine_pcr6(m2)
        conj = m1.combine_conjunctive(m2)
        assert pcr6 == conj

    def test_normalized(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        pcr6 = m1.combine_pcr6(m2)
        assert pcr6.is_normalized()

    def test_no_empty_set_mass(self):
        """PCR6 redistributes all conflict - no mass on empty set."""
        m1 = MassFunction(FRAME, {0b001: 0.6, 0b010: 0.4})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b100: 0.7})
        pcr6 = m1.combine_pcr6(m2)
        assert pcr6[0] == 0.0

    def test_hand_computed(self):
        """Verify PCR6 against hand-computed values on 2-element frame."""
        m1 = MassFunction(["a", "b"], {0b01: 0.6, 0b10: 0.4})
        m2 = MassFunction(["a", "b"], {0b01: 0.3, 0b10: 0.7})
        pcr6 = m1.combine_pcr6(m2)
        # Conjunctive part:
        #   {a}∩{a}={a}: 0.18, {b}∩{b}={b}: 0.28
        # Conflict pairs:
        #   {a}∩{b}=∅: conflict=0.6*0.7=0.42 → to {a}: 0.42*0.6/1.3, to {b}: 0.42*0.7/1.3
        #   {b}∩{a}=∅: conflict=0.4*0.3=0.12 → to {b}: 0.12*0.4/0.7, to {a}: 0.12*0.3/0.7
        to_a = 0.18 + 0.42 * 0.6 / 1.3 + 0.12 * 0.3 / 0.7
        to_b = 0.28 + 0.42 * 0.7 / 1.3 + 0.12 * 0.4 / 0.7
        assert pcr6[{"a"}] == pytest.approx(to_a)
        assert pcr6[{"b"}] == pytest.approx(to_b)
        assert to_a + to_b == pytest.approx(1.0)

    def test_standalone_matches_method(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert combine_pcr6(m1, m2) == m1.combine_pcr6(m2)

    def test_commutativity(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b010: 0.2, 0b111: 0.4})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b100: 0.3, 0b111: 0.4})
        assert m1.combine_pcr6(m2) == m2.combine_pcr6(m1)


# ── Murphy's average combination ──────────────────────────────────────────


class TestMurphy:
    def test_two_sources(self):
        m1 = MassFunction(FRAME, {0b001: 0.6, 0b111: 0.4})
        m2 = MassFunction(FRAME, {0b010: 0.6, 0b111: 0.4})
        result = combine_murphy(m1, m2)
        assert result.is_normalized()

    def test_identical_sources(self):
        """Murphy of identical sources = Dempster of source with itself."""
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        murphy = combine_murphy(m, m)
        # Average of m with m = m, then Dempster once
        dempster = m.combine_dempster(m)
        assert murphy == dempster

    def test_three_sources(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        m3 = MassFunction(FRAME, {0b100: 0.5, 0b111: 0.5})
        result = combine_murphy(m1, m2, m3)
        assert result.is_normalized()

    def test_too_few_raises(self):
        m = MassFunction(FRAME, {0b111: 1.0})
        with pytest.raises(ValueError, match="at least two"):
            combine_murphy(m)

    def test_incompatible_raises(self):
        m1 = MassFunction(["a", "b"], {0b11: 1.0})
        m2 = MassFunction(["x", "y"], {0b11: 1.0})
        with pytest.raises(ValueError, match="incompatible"):
            combine_murphy(m1, m2)


# ── Cautious rule ─────────────────────────────────────────────────────────


class TestCautious:
    def test_idempotent(self):
        """Cautious rule is idempotent: m ⊕_cautious m = m."""
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        result = m.combine_cautious(m)
        # Should be approximately equal to m
        for mask, mass in m.focal_elements().items():
            assert result[mask] == pytest.approx(mass, abs=1e-6), (
                f"idempotent failed at mask {mask}: {result[mask]} != {mass}"
            )

    def test_vacuous_neutral(self):
        """Cautious combination with vacuous gives back the original."""
        m = MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})
        v = MassFunction.vacuous(FRAME)
        result = m.combine_cautious(v)
        for mask, mass in m.focal_elements().items():
            assert result[mask] == pytest.approx(mass, abs=1e-6)

    def test_normalized(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b010: 0.2, 0b111: 0.8})
        result = m1.combine_cautious(m2)
        assert result.total_mass() == pytest.approx(1.0, abs=1e-6)

    def test_standalone_matches_method(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b010: 0.2, 0b111: 0.8})
        assert combine_cautious(m1, m2) == m1.combine_cautious(m2)

    def test_dogmatic_raises(self):
        """Dogmatic mass (all on singleton) has zero commonalities → error."""
        m1 = MassFunction.certain(FRAME, "a")
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        with pytest.raises(ValueError, match="non-dogmatic"):
            m1.combine_cautious(m2)

    def test_simple_support(self):
        """Two simple support functions on different hypotheses."""
        # Simple support: m({a})=0.4, m(Ω)=0.6
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        # Simple support: m({b})=0.3, m(Ω)=0.7
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        result = m1.combine_cautious(m2)
        assert result.is_normalized()
        # Should have mass on both {a} and {b}
        assert result[0b001] > 0
        assert result[0b010] > 0


# ── Bold rule ─────────────────────────────────────────────────────────────


class TestBold:
    def test_idempotent(self):
        """Bold rule is idempotent: m ⊕_bold m = m."""
        # Need a mass function where all Bel(A)>0 for proper non-empty subsets
        # This requires mass on all singletons
        m = MassFunction(FRAME, {0b001: 0.2, 0b010: 0.2, 0b100: 0.2, 0b111: 0.4})
        result = m.combine_bold(m)
        for mask, mass in m.focal_elements().items():
            assert result[mask] == pytest.approx(mass, abs=1e-6), (
                f"idempotent failed at mask {mask}: {result[mask]} != {mass}"
            )

    def test_normalized(self):
        m = MassFunction(FRAME, {0b001: 0.2, 0b010: 0.2, 0b100: 0.2, 0b111: 0.4})
        result = m.combine_bold(m)
        assert result.total_mass() == pytest.approx(1.0, abs=1e-6)

    def test_standalone_matches_method(self):
        # Use a 2-element frame: the bold rule always yields a proper
        # belief function on 2-element frames (no intermediate-set
        # constraint violations).
        frame2 = ["a", "b"]
        m1 = MassFunction(frame2, {0b01: 0.4, 0b10: 0.3, 0b11: 0.3})
        m2 = MassFunction(frame2, {0b01: 0.2, 0b10: 0.5, 0b11: 0.3})
        assert combine_bold(m1, m2) == m1.combine_bold(m2)

    def test_zero_belief_raises(self):
        """Mass function with zero Bel for some proper subset → error."""
        # m({a})=1, Bel({b})=0
        m1 = MassFunction.certain(FRAME, "a")
        m2 = MassFunction(FRAME, {0b001: 0.2, 0b010: 0.2, 0b100: 0.2, 0b111: 0.4})
        with pytest.raises(ValueError, match="non-vacuous"):
            m1.combine_bold(m2)


# ── Multi-source fusion ──────────────────────────────────────────────────


class TestCombineMultiple:
    def test_dempster_two(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        assert combine_multiple([m1, m2], "dempster") == m1.combine_dempster(m2)

    def test_dempster_three(self):
        m1 = MassFunction(FRAME, {0b001: 0.3, 0b111: 0.7})
        m2 = MassFunction(FRAME, {0b010: 0.2, 0b111: 0.8})
        m3 = MassFunction(FRAME, {0b100: 0.1, 0b111: 0.9})
        result = combine_multiple([m1, m2, m3], "dempster")
        expected = (m1 & m2) & m3
        assert result == expected

    def test_disjunctive(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        result = combine_multiple([m1, m2], "disjunctive")
        assert result == m1.combine_disjunctive(m2)

    def test_murphy(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        result = combine_multiple([m1, m2], "murphy")
        assert result == combine_murphy(m1, m2)

    def test_dubois_prade(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        result = combine_multiple([m1, m2], "dubois-prade")
        assert result == m1.combine_dubois_prade(m2)

    def test_pcr6_multi(self):
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        result = combine_multiple([m1, m2], "pcr6")
        assert result == m1.combine_pcr6(m2)

    def test_unknown_rule_raises(self):
        m1 = MassFunction(FRAME, {0b111: 1.0})
        m2 = MassFunction(FRAME, {0b111: 1.0})
        with pytest.raises(ValueError, match="unknown rule"):
            combine_multiple([m1, m2], "nonexistent")

    def test_too_few_raises(self):
        m = MassFunction(FRAME, {0b111: 1.0})
        with pytest.raises(ValueError, match="at least two"):
            combine_multiple([m], "dempster")

    def test_rule_name_normalization(self):
        """Underscores, spaces, and case are normalized."""
        m1 = MassFunction(FRAME, {0b001: 0.5, 0b111: 0.5})
        m2 = MassFunction(FRAME, {0b010: 0.5, 0b111: 0.5})
        r1 = combine_multiple([m1, m2], "Dubois-Prade")
        r2 = combine_multiple([m1, m2], "dubois_prade")
        r3 = combine_multiple([m1, m2], "DUBOIS PRADE")
        assert r1 == r2 == r3

    def test_conjunctive(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        result = combine_multiple([m1, m2], "conjunctive")
        assert result == m1.combine_conjunctive(m2)

    def test_yager(self):
        m1 = MassFunction(FRAME, {0b001: 0.4, 0b111: 0.6})
        m2 = MassFunction(FRAME, {0b010: 0.3, 0b111: 0.7})
        result = combine_multiple([m1, m2], "yager")
        assert result == m1.combine_yager(m2)
