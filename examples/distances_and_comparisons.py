"""Distances and comparisons between mass functions.

Demonstrates all 12 measures from pybelief.distances.
"""

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

frame = ["a", "b", "c"]

# ── 1. Define three mass functions ────────────────────────────────────

m1 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.3,
        frozenset({"a", "b"}): 0.2,
        frozenset({"a", "b", "c"}): 0.5,
    },
)

m2 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"b"}): 0.5,
        frozenset({"a", "b"}): 0.2,
        frozenset({"a", "b", "c"}): 0.3,
    },
)

m3 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.35,
        frozenset({"a", "b"}): 0.15,
        frozenset({"a", "b", "c"}): 0.5,
    },
)

print("m1:", m1)
print()
print("m2:", m2)
print()
print("m3:", m3)
print()

# ── 2. Metric distances ──────────────────────────────────────────────

print("=== Metric distances ===")
print(f"  Jousselme(m1, m2)  = {jousselme(m1, m2):.4f}")
print(f"  Jousselme(m1, m3)  = {jousselme(m1, m3):.4f}")
print(f"  Jousselme(m2, m3)  = {jousselme(m2, m3):.4f}")
print()
print(f"  Euclidean(m1, m2)  = {euclidean(m1, m2):.4f}")
print(f"  Euclidean(m1, m3)  = {euclidean(m1, m3):.4f}")
print()

# Verify triangle inequality: d(m1,m2) <= d(m1,m3) + d(m3,m2)
d12 = jousselme(m1, m2)
d13 = jousselme(m1, m3)
d32 = jousselme(m3, m2)
print(f"  Triangle inequality: {d12:.4f} <= {d13:.4f} + {d32:.4f} = {d13 + d32:.4f}  ({'OK' if d12 <= d13 + d32 + 1e-9 else 'FAIL'})")
print()

# ── 3. Pignistic-based distances ─────────────────────────────────────

print("=== Pignistic-based distances ===")
print(f"  Tessem (L-inf)     = {tessem(m1, m2):.4f}")
print(f"  Pignistic L1       = {pignistic_l1(m1, m2):.4f}")
print(f"  Pignistic L2       = {pignistic_l2(m1, m2):.4f}")
print(f"  Bhattacharyya      = {bhattacharyya(m1, m2):.4f}")
print()

# ── 4. Cosine similarity ─────────────────────────────────────────────

print("=== Cosine similarity ===")
print(f"  sim(m1, m2)        = {cosine_similarity(m1, m2):.4f}")
print(f"  sim(m1, m3)        = {cosine_similarity(m1, m3):.4f}  (m3 is closer to m1)")
print(f"  dist(m1, m2)       = {cosine_distance(m1, m2):.4f}")
print()

# ── 5. Conflict-based distance ───────────────────────────────────────

print("=== Conflict-based ===")
print(f"  K(m1, m2)          = {m1.conflict(m2):.4f}")
print(f"  d_K(m1, m2)        = {conflict_distance(m1, m2):.4f}")
print()

# Total conflict example
m_a = MassFunction.certain(frame, "a")
m_b = MassFunction.certain(frame, "b")
print(f"  K(certain_a, certain_b)   = {m_a.conflict(m_b):.4f}")
print(f"  d_K(certain_a, certain_b) = {conflict_distance(m_a, m_b)}")
print()

# ── 6. Deng relative entropy (asymmetric) ────────────────────────────

print("=== Deng relative entropy (asymmetric) ===")
d12 = deng_relative_entropy(m1, m2)
d21 = deng_relative_entropy(m2, m1)
print(f"  D(m1 || m2)        = {d12:.4f}")
print(f"  D(m2 || m1)        = {d21:.4f}")
print(f"  Asymmetric: {d12:.4f} != {d21:.4f}")
print()

# For Bayesian masses, this equals KL divergence
mb1 = MassFunction.from_bayesian(["a", "b"], {"a": 0.7, "b": 0.3})
mb2 = MassFunction.from_bayesian(["a", "b"], {"a": 0.4, "b": 0.6})
print(f"  Bayesian D(mb1 || mb2) = {deng_relative_entropy(mb1, mb2):.4f}  (= KL divergence)")
print()

# ── 7. Inclusion degree (asymmetric) ─────────────────────────────────

print("=== Inclusion degree (asymmetric) ===")
print(f"  Inc(m1, m2)        = {inclusion_degree(m1, m2):.4f}")
print(f"  Inc(m2, m1)        = {inclusion_degree(m2, m1):.4f}")
print()
# Vacuous always fully includes
m_vac = MassFunction.vacuous(frame)
print(f"  Inc(m1, vacuous)   = {inclusion_degree(m1, m_vac):.4f}  (always 1.0)")
print()

# ── 8. Auto-conflict ─────────────────────────────────────────────────

print("=== Auto-conflict ===")
print(f"  K_auto(m1)         = {auto_conflict(m1):.4f}")
print(f"  K_auto(m2)         = {auto_conflict(m2):.4f}")
print(f"  K_auto(vacuous)    = {auto_conflict(m_vac):.4f}")
print()

# High internal conflict: mass split between disjoint singletons
m_split = MassFunction(frame, {0b001: 0.5, 0b010: 0.5})
print(f"  K_auto(split a/b)  = {auto_conflict(m_split):.4f}  (high conflict)")
print()

# ── 9. Comparing all distances at a glance ───────────────────────────

print("=== Summary: m1 vs m2 ===")
print(f"  {'Jousselme':<25s} {jousselme(m1, m2):.6f}")
print(f"  {'Euclidean':<25s} {euclidean(m1, m2):.6f}")
print(f"  {'Bhattacharyya':<25s} {bhattacharyya(m1, m2):.6f}")
print(f"  {'Tessem (L-inf BetP)':<25s} {tessem(m1, m2):.6f}")
print(f"  {'Pignistic L1':<25s} {pignistic_l1(m1, m2):.6f}")
print(f"  {'Pignistic L2':<25s} {pignistic_l2(m1, m2):.6f}")
print(f"  {'Conflict distance':<25s} {conflict_distance(m1, m2):.6f}")
print(f"  {'Cosine distance':<25s} {cosine_distance(m1, m2):.6f}")
print(f"  {'Deng D(m1||m2)':<25s} {deng_relative_entropy(m1, m2):.6f}")
print(f"  {'Inclusion Inc(m1,m2)':<25s} {inclusion_degree(m1, m2):.6f}")
print(f"  {'Auto-conflict m1':<25s} {auto_conflict(m1):.6f}")
