"""Decision-making under belief functions with pybelief."""

from pybelief import MassFunction, DecisionProblem

# ── 1. Define a decision problem ─────────────────────────────────────

# States of nature
states = ["s1", "s2", "s3"]

# Evidence (mass function on the states)
m = MassFunction(
    states,
    named_focal_elements={
        frozenset({"s1"}): 0.3,
        frozenset({"s1", "s2"}): 0.3,
        frozenset({"s1", "s2", "s3"}): 0.4,
    },
)

# Utility matrix: what each action yields in each state
utilities = {
    "a1": {"s1": 10, "s2": 5, "s3": 2},
    "a2": {"s1": 3, "s2": 8, "s3": 9},
}

dp = DecisionProblem(m, utilities)
print(f"Actions: {dp.actions}")
print(f"States:  {dp.states}")
print()

# ── 2. Choquet expected-utility intervals ────────────────────────────

print("=== Expected-utility intervals ===")
for action in dp.actions:
    lo, hi = dp.expectation_interval(action)
    print(f"  {action}: [{lo:.4f}, {hi:.4f}]")
print()

# ── 3. Ranking criteria ─────────────────────────────────────────────

print("=== Maximin (pessimistic) ===")
for action, score in dp.maximin():
    print(f"  {action}: {score:.4f}")
print()

print("=== Maximax (optimistic) ===")
for action, score in dp.maximax():
    print(f"  {action}: {score:.4f}")
print()

print("=== Hurwicz (alpha=0.5) ===")
for action, score in dp.hurwicz(alpha=0.5):
    print(f"  {action}: {score:.4f}")
print()

# Different optimism levels
print("=== Hurwicz sensitivity ===")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    winner, score = dp.hurwicz(alpha=alpha)[0]
    print(f"  alpha={alpha:.2f} → best={winner} (score={score:.4f})")
print()

# ── 4. Set-valued criteria ──────────────────────────────────────────

print("=== Interval dominance ===")
print(f"  Surviving actions: {dp.interval_dominance()}")
print()

print("=== Maximality ===")
print(f"  Surviving actions: {dp.maximality()}")
print()

try:
    print("=== E-admissibility ===")
    print(f"  Surviving actions: {dp.e_admissibility()}")
    print()
except ImportError:
    print("  (scipy required for E-admissibility)")
    print()

# ── 5. Summary ──────────────────────────────────────────────────────

print("=== Full summary ===")
summary = dp.summary()
for key, value in summary.items():
    print(f"  {key}: {value}")
print()

# ── 6. A problem with clear dominance ───────────────────────────────

print("=" * 60)
print("=== Dominated action example ===")
m2 = MassFunction(
    ["s1", "s2"],
    named_focal_elements={
        frozenset({"s1"}): 0.5,
        frozenset({"s1", "s2"}): 0.5,
    },
)
dp2 = DecisionProblem(
    m2,
    {
        "good": {"s1": 10, "s2": 10},
        "bad": {"s1": 1, "s2": 1},
    },
)
print(f"  Maximin winner:          {dp2.maximin()[0][0]}")
print(f"  Interval dominance:      {dp2.interval_dominance()}")
print(f"  Maximality:              {dp2.maximality()}")
try:
    print(f"  E-admissibility:         {dp2.e_admissibility()}")
except ImportError:
    pass
