"""Tests for pybelief.decision - decision-making under belief functions."""

import pytest
from pybelief import MassFunction, DecisionProblem


STATES = ["s1", "s2", "s3"]


@pytest.fixture()
def simple_problem():
    """A simple decision problem with 2 actions and 3 states."""
    m = MassFunction(
        STATES,
        named_focal_elements={
            frozenset({"s1"}): 0.3,
            frozenset({"s1", "s2"}): 0.3,
            frozenset({"s1", "s2", "s3"}): 0.4,
        },
    )
    utilities = {
        "a1": {"s1": 10, "s2": 5, "s3": 2},
        "a2": {"s1": 3, "s2": 8, "s3": 9},
    }
    return DecisionProblem(m, utilities)


@pytest.fixture()
def dominated_problem():
    """A problem where one action clearly dominates."""
    m = MassFunction(
        ["s1", "s2"],
        named_focal_elements={
            frozenset({"s1"}): 0.5,
            frozenset({"s1", "s2"}): 0.5,
        },
    )
    utilities = {
        "good": {"s1": 10, "s2": 10},
        "bad": {"s1": 1, "s2": 1},
    }
    return DecisionProblem(m, utilities)


# ── Construction ──────────────────────────────────────────────────────


class TestConstruction:
    def test_basic(self, simple_problem):
        dp = simple_problem
        assert dp.actions == ("a1", "a2")
        assert dp.states == tuple(STATES)

    def test_empty_actions_raises(self):
        m = MassFunction.vacuous(STATES)
        with pytest.raises(ValueError, match="at least one"):
            DecisionProblem(m, {})

    def test_missing_state_raises(self):
        m = MassFunction.vacuous(STATES)
        with pytest.raises(ValueError, match="missing states"):
            DecisionProblem(m, {"a": {"s1": 1, "s2": 2}})

    def test_unknown_state_raises(self):
        m = MassFunction.vacuous(STATES)
        with pytest.raises(ValueError, match="unknown states"):
            DecisionProblem(m, {"a": {"s1": 1, "s2": 2, "s3": 3, "s4": 4}})


# ── Choquet integrals ─────────────────────────────────────────────────


class TestChoquet:
    def test_lower_expectation(self, simple_problem):
        dp = simple_problem
        # a1: focal {s1}→10 (mass 0.3), {s1,s2}→min(10,5)=5 (mass 0.3),
        #     {s1,s2,s3}→min(10,5,2)=2 (mass 0.4)
        # E_*(a1) = 0.3*10 + 0.3*5 + 0.4*2 = 3 + 1.5 + 0.8 = 5.3
        assert dp.lower_expectation("a1") == pytest.approx(5.3)

    def test_upper_expectation(self, simple_problem):
        dp = simple_problem
        # a1: focal {s1}→10 (mass 0.3), {s1,s2}→max(10,5)=10 (mass 0.3),
        #     {s1,s2,s3}→max(10,5,2)=10 (mass 0.4)
        # E*(a1) = 0.3*10 + 0.3*10 + 0.4*10 = 10
        assert dp.upper_expectation("a1") == pytest.approx(10.0)

    def test_interval(self, simple_problem):
        lo, hi = simple_problem.expectation_interval("a1")
        assert lo == pytest.approx(5.3)
        assert hi == pytest.approx(10.0)

    def test_bayesian_lower_equals_upper(self):
        """For a Bayesian mass, lower = upper = standard expected utility."""
        m = MassFunction.from_bayesian(
            ["s1", "s2"], {"s1": 0.6, "s2": 0.4}
        )
        dp = DecisionProblem(m, {"a": {"s1": 10, "s2": 5}})
        assert dp.lower_expectation("a") == pytest.approx(8.0)
        assert dp.upper_expectation("a") == pytest.approx(8.0)


# ── Maximin / Maximax ─────────────────────────────────────────────────


class TestMaximinMaximax:
    def test_maximin(self, simple_problem):
        ranked = simple_problem.maximin()
        # a2 lower: 0.3*3 + 0.3*3 + 0.4*3 = 3? No:
        # a2: {s1}→3 (0.3), {s1,s2}→min(3,8)=3 (0.3), {s1,s2,s3}→min(3,8,9)=3 (0.4)
        # E_*(a2) = 0.3*3 + 0.3*3 + 0.4*3 = 3.0
        # E_*(a1) = 5.3
        assert ranked[0][0] == "a1"
        assert ranked[0][1] == pytest.approx(5.3)

    def test_maximax(self, simple_problem):
        ranked = simple_problem.maximax()
        # E*(a1) = 10, E*(a2) = 0.3*3 + 0.3*8 + 0.4*9 = 0.9+2.4+3.6=6.9
        # Wait: E*(a2): {s1}→3, {s1,s2}→max(3,8)=8, {s1,s2,s3}→max(3,8,9)=9
        # = 0.3*3 + 0.3*8 + 0.4*9 = 0.9 + 2.4 + 3.6 = 6.9
        assert ranked[0][0] == "a1"
        assert ranked[0][1] == pytest.approx(10.0)

    def test_dominated_maximin(self, dominated_problem):
        ranked = dominated_problem.maximin()
        assert ranked[0][0] == "good"

    def test_dominated_maximax(self, dominated_problem):
        ranked = dominated_problem.maximax()
        assert ranked[0][0] == "good"


# ── Hurwicz ───────────────────────────────────────────────────────────


class TestHurwicz:
    def test_alpha_zero_is_maximin(self, simple_problem):
        h = simple_problem.hurwicz(alpha=0.0)
        mm = simple_problem.maximin()
        assert h[0][0] == mm[0][0]
        assert h[0][1] == pytest.approx(mm[0][1])

    def test_alpha_one_is_maximax(self, simple_problem):
        h = simple_problem.hurwicz(alpha=1.0)
        mx = simple_problem.maximax()
        assert h[0][0] == mx[0][0]
        assert h[0][1] == pytest.approx(mx[0][1])

    def test_alpha_half(self, simple_problem):
        h = simple_problem.hurwicz(alpha=0.5)
        # a1: 0.5*10 + 0.5*5.3 = 7.65
        # a2: 0.5*6.9 + 0.5*3.0 = 4.95
        assert h[0][0] == "a1"
        assert h[0][1] == pytest.approx(7.65)

    def test_invalid_alpha(self, simple_problem):
        with pytest.raises(ValueError):
            simple_problem.hurwicz(alpha=1.5)


# ── Interval dominance ────────────────────────────────────────────────


class TestIntervalDominance:
    def test_no_dominance(self, simple_problem):
        # Neither dominates the other by interval
        surviving = simple_problem.interval_dominance()
        assert surviving == {"a1", "a2"}

    def test_clear_dominance(self, dominated_problem):
        surviving = dominated_problem.interval_dominance()
        assert "good" in surviving
        assert "bad" not in surviving


# ── Maximality ────────────────────────────────────────────────────────


class TestMaximality:
    def test_no_dominance(self, simple_problem):
        surviving = simple_problem.maximality()
        assert surviving == {"a1", "a2"}

    def test_clear_dominance(self, dominated_problem):
        surviving = dominated_problem.maximality()
        assert surviving == {"good"}

    def test_superset_of_e_admissibility(self, simple_problem):
        """E-admissibility contains fewer or equal elements than Maximality."""
        maximal = simple_problem.maximality()
        try:
            e_adm = simple_problem.e_admissibility()
            assert e_adm <= maximal
        except ImportError:
            pytest.skip("scipy not available")


# ── E-admissibility ──────────────────────────────────────────────────


class TestEAdmissibility:
    def test_basic(self, simple_problem):
        try:
            adm = simple_problem.e_admissibility()
        except ImportError:
            pytest.skip("scipy not available")
        assert isinstance(adm, set)
        assert len(adm) >= 1

    def test_clear_dominance(self, dominated_problem):
        try:
            adm = dominated_problem.e_admissibility()
        except ImportError:
            pytest.skip("scipy not available")
        assert adm == {"good"}

    def test_hierarchy(self):
        """e_admissibility ⊆ maximality ⊆ interval_dominance."""
        m = MassFunction(
            ["s1", "s2", "s3"],
            named_focal_elements={
                frozenset({"s1"}): 0.2,
                frozenset({"s2", "s3"}): 0.3,
                frozenset({"s1", "s2", "s3"}): 0.5,
            },
        )
        utils = {
            "a1": {"s1": 10, "s2": 2, "s3": 1},
            "a2": {"s1": 1, "s2": 8, "s3": 9},
            "a3": {"s1": 5, "s2": 5, "s3": 5},
        }
        dp = DecisionProblem(m, utils)
        intdom = dp.interval_dominance()
        maximal = dp.maximality()
        try:
            e_adm = dp.e_admissibility()
        except ImportError:
            pytest.skip("scipy not available")
        assert e_adm <= maximal
        assert maximal <= intdom


# ── Summary ───────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_keys(self, simple_problem):
        s = simple_problem.summary()
        assert "intervals" in s
        assert "maximin" in s
        assert "maximax" in s
        assert "hurwicz_0.5" in s
        assert "interval_dominance" in s
        assert "maximality" in s
        assert "e_admissibility" in s
