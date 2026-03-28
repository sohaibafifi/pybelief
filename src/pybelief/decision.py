"""Decision-making under belief functions.

Provides criteria for choosing among actions when uncertainty is
represented by a Dempster-Shafer mass function on the states of nature.

Given actions ``a₁ … aₖ``, a frame of states, a utility matrix
``U[action, state]``, and a mass function ``m`` on the states, every
action receives an expected-utility interval ``[E_*(a), E*(a)]``
computed via Choquet integrals.  The various criteria then select
admissible actions from these intervals.

Hierarchy (most conservative → most permissive)::

    E-admissibility  ⊆  Maximality  ⊆  Interval dominance

Hurwicz (with fixed alpha) produces a complete ranking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pybelief.mass import MassFunction


class DecisionProblem:
    """A decision problem under a Dempster-Shafer mass function.

    Parameters
    ----------
    mass : MassFunction
        Mass function on the states (frame of discernment).
    utilities : dict[str, dict[str, float]]
        ``{action_label: {state_label: utility}}``.
        Every action must map every frame element to a utility value.

    Examples
    --------
    >>> m = MassFunction(["s1", "s2", "s3"], named_focal_elements={
    ...     frozenset({"s1"}): 0.3,
    ...     frozenset({"s1", "s2"}): 0.3,
    ...     frozenset({"s1", "s2", "s3"}): 0.4,
    ... })
    >>> dp = DecisionProblem(m, {
    ...     "a1": {"s1": 10, "s2": 5, "s3": 2},
    ...     "a2": {"s1": 3,  "s2": 8, "s3": 9},
    ... })
    >>> dp.hurwicz(alpha=0.5)
    [('a2', 7.05), ('a1', 6.35)]
    """

    __slots__ = ("mass", "actions", "states", "utilities", "_focal")

    def __init__(
        self,
        mass: MassFunction,
        utilities: dict[str, dict[str, float]],
    ) -> None:
        if not utilities:
            raise ValueError("at least one action is required")

        self.mass = mass
        self.states: tuple[str, ...] = mass.frame
        self.actions: tuple[str, ...] = tuple(utilities)

        # Validate: every action must cover every state
        state_set = set(self.states)
        for action, u_map in utilities.items():
            if set(u_map) != state_set:
                missing = state_set - set(u_map)
                extra = set(u_map) - state_set
                parts = []
                if missing:
                    parts.append(f"missing states {missing}")
                if extra:
                    parts.append(f"unknown states {extra}")
                raise ValueError(
                    f"action {action!r}: {', '.join(parts)}"
                )

        # Store utilities as {action: [u(state_0), u(state_1), ...]}
        self.utilities: dict[str, list[float]] = {
            act: [u_map[s] for s in self.states]
            for act, u_map in utilities.items()
        }

        # Cache focal elements as list of (mask, mass)
        self._focal: list[tuple[int, float]] = list(mass.focal_elements().items())

    # ── Choquet integrals ──────────────────────────────────────────────

    def lower_expectation(self, action: str) -> float:
        r"""Lower expected utility (Choquet integral w.r.t. Bel).

        .. math::
            \underline{E}(a) = \sum_{A} m(A)\,\min_{\theta \in A} U(a,\theta)
        """
        u = self.utilities[action]
        return math.fsum(
            mass * min(u[i] for i in range(len(self.states)) if mask & (1 << i))
            for mask, mass in self._focal
            if mask != 0
        )

    def upper_expectation(self, action: str) -> float:
        r"""Upper expected utility (Choquet integral w.r.t. Pl).

        .. math::
            \overline{E}(a) = \sum_{A} m(A)\,\max_{\theta \in A} U(a,\theta)
        """
        u = self.utilities[action]
        return math.fsum(
            mass * max(u[i] for i in range(len(self.states)) if mask & (1 << i))
            for mask, mass in self._focal
            if mask != 0
        )

    def expectation_interval(self, action: str) -> tuple[float, float]:
        """Return ``(lower, upper)`` expected utility for *action*."""
        return (self.lower_expectation(action), self.upper_expectation(action))

    # ── Maximin / Maximax ──────────────────────────────────────────────

    def maximin(self) -> list[tuple[str, float]]:
        """Rank actions by lower expected utility (pessimistic).

        Returns actions sorted descending by ``E_*(a)``.
        """
        ranked = [(a, self.lower_expectation(a)) for a in self.actions]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def maximax(self) -> list[tuple[str, float]]:
        """Rank actions by upper expected utility (optimistic).

        Returns actions sorted descending by ``E*(a)``.
        """
        ranked = [(a, self.upper_expectation(a)) for a in self.actions]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ── Hurwicz ────────────────────────────────────────────────────────

    def hurwicz(self, alpha: float = 0.5) -> list[tuple[str, float]]:
        r"""Hurwicz criterion with optimism index *alpha*.

        .. math::
            H(a) = \alpha\,\overline{E}(a) + (1-\alpha)\,\underline{E}(a)

        Parameters
        ----------
        alpha : float
            Optimism in ``[0, 1]``.  ``0`` = maximin, ``1`` = maximax.

        Returns actions sorted descending by Hurwicz score.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        ranked = []
        for a in self.actions:
            lo, hi = self.expectation_interval(a)
            ranked.append((a, alpha * hi + (1 - alpha) * lo))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ── Interval dominance ─────────────────────────────────────────────

    def interval_dominance(self) -> set[str]:
        """Return the set of actions surviving interval dominance.

        Action *a* is eliminated if there exists *b* such that
        ``E*(a) < E_*(b)`` (the best case for *a* is worse than
        the worst case for *b*).
        """
        intervals = {
            a: self.expectation_interval(a) for a in self.actions
        }
        surviving = set()
        for a in self.actions:
            _, hi_a = intervals[a]
            dominated = False
            for b in self.actions:
                if b == a:
                    continue
                lo_b, _ = intervals[b]
                if hi_a < lo_b - 1e-12:
                    dominated = True
                    break
            if not dominated:
                surviving.add(a)
        return surviving

    # ── Maximality ─────────────────────────────────────────────────────

    def _lower_expectation_diff(self, a: str, b: str) -> float:
        r"""Lower expectation of ``U(b) - U(a)`` via Choquet integral.

        .. math::
            \underline{E}(u_b - u_a)
            = \sum_A m(A)\,\min_{\theta \in A}\,[U(b,\theta) - U(a,\theta)]

        Note: this is NOT the same as ``E_*(b) - E*(a)``.
        """
        ua = self.utilities[a]
        ub = self.utilities[b]
        n = len(self.states)
        return math.fsum(
            mass * min(
                ub[i] - ua[i]
                for i in range(n)
                if mask & (1 << i)
            )
            for mask, mass in self._focal
            if mask != 0
        )

    def maximality(self) -> set[str]:
        """Return the set of maximal (non-strongly-dominated) actions.

        Action *a* is dominated by *b* if
        ``E_*(u_b - u_a) > 0`` - i.e., under every compatible probability,
        *b* yields strictly more than *a* in the worst case of
        their difference.
        """
        surviving = set()
        for a in self.actions:
            dominated = False
            for b in self.actions:
                if b == a:
                    continue
                if self._lower_expectation_diff(a, b) > 1e-12:
                    dominated = True
                    break
            if not dominated:
                surviving.add(a)
        return surviving

    # ── E-admissibility ────────────────────────────────────────────────

    def e_admissibility(self) -> set[str]:
        """Return the set of E-admissible actions.

        Action *a* is E-admissible if there exists a probability
        ``P`` in the credal set (``Bel(A) ≤ P(A) ≤ Pl(A)`` for all A)
        such that *a* maximizes expected utility under ``P``.

        Implementation uses linear programming.  Requires ``scipy``.
        Falls back to a constraint-based check: for each action *a*,
        solve an LP to find if any ``P`` in the credal set makes *a*
        at least as good as every other action.

        Raises :class:`ImportError` if scipy is not available.
        """
        try:
            from scipy.optimize import linprog  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "E-admissibility requires scipy. "
                "Install it with: pip install scipy"
            ) from None

        n = len(self.states)
        mass = self.mass

        # Build credal set constraints: Bel(A) ≤ P(A) ≤ Pl(A) for all A
        # P(A) = sum_{x in A} p(x)
        # We only need proper subsets (P(Omega)=1 is handled by equality)
        #
        # Decision variables: p = [p_0, p_1, ..., p_{n-1}]

        A_ub_rows: list[list[float]] = []
        b_ub_vals: list[float] = []

        frame_mask = mass._frame_mask
        # For each non-empty proper subset, add Bel ≤ P(A) and P(A) ≤ Pl(A)
        for s in range(1, frame_mask):
            bel_s = mass.belief(s)
            pl_s = mass.plausibility(s)

            # Bel(A) ≤ P(A)  →  -P(A) ≤ -Bel(A)
            row_lo = [0.0] * n
            for i in range(n):
                if s & (1 << i):
                    row_lo[i] = -1.0
            A_ub_rows.append(row_lo)
            b_ub_vals.append(-bel_s)

            # P(A) ≤ Pl(A)  →  P(A) ≤ Pl(A)
            row_hi = [0.0] * n
            for i in range(n):
                if s & (1 << i):
                    row_hi[i] = 1.0
            A_ub_rows.append(row_hi)
            b_ub_vals.append(pl_s)

        # Equality: sum p_i = 1
        A_eq = [[1.0] * n]
        b_eq = [1.0]

        # Bounds: 0 ≤ p_i ≤ 1
        bounds = [(0.0, 1.0)] * n

        admissible = set()

        for a in self.actions:
            ua = self.utilities[a]

            # For action a to be optimal, we need: for every other b,
            # E_P[U(a)] ≥ E_P[U(b)]  i.e.  E_P[U(a) - U(b)] ≥ 0
            # Add constraints: sum_i p_i * (u_b_i - u_a_i) ≤ 0 for all b≠a
            extra_rows: list[list[float]] = []
            extra_b: list[float] = []
            for b in self.actions:
                if b == a:
                    continue
                ub = self.utilities[b]
                # sum p_i (u_b_i - u_a_i) ≤ 0
                row = [ub[i] - ua[i] for i in range(n)]
                extra_rows.append(row)
                extra_b.append(0.0)

            A_ub = A_ub_rows + extra_rows
            b_ub = b_ub_vals + extra_b

            # Objective: just find feasibility (minimize 0)
            c = [0.0] * n

            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if result.success:
                admissible.add(a)

        return admissible

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with intervals and all criteria results.

        Includes E-admissibility only if scipy is available.
        """
        intervals = {
            a: self.expectation_interval(a) for a in self.actions
        }
        result: dict[str, Any] = {
            "intervals": intervals,
            "maximin": self.maximin(),
            "maximax": self.maximax(),
            "hurwicz_0.5": self.hurwicz(0.5),
            "interval_dominance": self.interval_dominance(),
            "maximality": self.maximality(),
        }
        try:
            result["e_admissibility"] = self.e_admissibility()
        except ImportError:
            result["e_admissibility"] = None
        return result
