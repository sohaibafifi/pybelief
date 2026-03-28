"""Combination rules - fusing evidence from multiple sources.

Demonstrates all combination rules available in pybelief, including
the new Phase 1 rules: disjunctive, Dubois-Prade, PCR6, Murphy,
cautious (Denœux), bold (Denœux), and multi-source fusion.
"""

from pybelief import (
    MassFunction,
    combine_murphy,
    combine_multiple,
    table,
)

# ── Setup: two conflicting sources on frame {a, b, c} ────────────────

frame = ["a", "b", "c"]

m1 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.5,
        frozenset({"a", "b"}): 0.2,
        frozenset({"a", "b", "c"}): 0.3,
    },
)

m2 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"b"}): 0.6,
        frozenset({"b", "c"}): 0.1,
        frozenset({"a", "b", "c"}): 0.3,
    },
)

print("Source 1:")
print(table(m1))
print()
print("Source 2:")
print(table(m2))
print()

# ── 1. Existing rules (for comparison) ───────────────────────────────

print("=" * 60)
print("=== Dempster's rule (m1 & m2) ===")
md = m1.combine_dempster(m2)
print(table(md))
print(f"  Conflict K = {m1.conflict(m2):.4f}")
print()

print("=== Conjunctive / TBM (m1 | m2) ===")
mc = m1.combine_conjunctive(m2)
print(table(mc, empty_set=True))
print()

print("=== Yager's rule ===")
my = m1.combine_yager(m2)
print(table(my))
print()

# ── 2. Disjunctive rule ─────────────────────────────────────────────

print("=" * 60)
print("=== Disjunctive rule ===")
print("  Uses union: m12(C) = Σ_{A∪B=C} m1(A)·m2(B)")
print("  (Smets 1990)")
mdsj = m1.combine_disjunctive(m2)
print(table(mdsj))
print()

# ── 3. Dubois-Prade rule ────────────────────────────────────────────

print("=" * 60)
print("=== Dubois-Prade rule ===")
print("  Conflict mass on A∩B=∅ goes to A∪B")
print("  (Dubois & Prade 1988)")
mdp = m1.combine_dubois_prade(m2)
print(table(mdp))
print()

# ── 4. PCR6 (Proportional Conflict Redistribution) ──────────────────

print("=" * 60)
print("=== PCR6 ===")
print("  Conflict redistributed proportionally to each hypothesis")
print("  (Smarandache & Dezert 2006)")
mpcr6 = m1.combine_pcr6(m2)
print(table(mpcr6))
print()

# ── 5. Murphy's average combination ─────────────────────────────────

print("=" * 60)
print("=== Murphy's average combination ===")
print("  Average mass functions, then apply Dempster (n-1) times")
print("  (Murphy 2000)")
mmurphy = combine_murphy(m1, m2)
print(table(mmurphy))
print()

# ── 6. Cautious rule (Denœux) ───────────────────────────────────────

print("=" * 60)
print("=== Cautious rule (Denœux) ===")
print("  Idempotent: m ⊕_cautious m = m")
print("  Uses min in weight-function domain")
print("  (Denœux 2008, Smets 1995)")

# Need non-dogmatic mass functions for cautious rule
m1c = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.4,
        frozenset({"a", "b", "c"}): 0.6,
    },
)
m2c = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"b"}): 0.3,
        frozenset({"a", "b", "c"}): 0.7,
    },
)
mcautious = m1c.combine_cautious(m2c)
print(table(mcautious))
print()

# Demonstrate idempotency
mcaut_idem = m1c.combine_cautious(m1c)
print("  Idempotency check (m1c ⊕ m1c == m1c):", mcaut_idem == m1c)
print()

# ── 7. Bold rule (Denœux) ───────────────────────────────────────────

print("=" * 60)
print("=== Bold rule (Denœux) ===")
print("  Disjunctive counterpart of cautious, uses max in weight domain")
print("  (Denœux 2008)")

# Bold rule needs Bel(A)>0 for all proper non-empty subsets
m1b = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.2,
        frozenset({"b"}): 0.2,
        frozenset({"c"}): 0.2,
        frozenset({"a", "b", "c"}): 0.4,
    },
)
m2b = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.1,
        frozenset({"b"}): 0.3,
        frozenset({"c"}): 0.1,
        frozenset({"a", "b", "c"}): 0.5,
    },
)
mbold = m1b.combine_bold(m2b)
print(table(mbold))
print()

# ── 8. Multi-source fusion  ──────────────────────────────────────────

print("=" * 60)
print("=== Multi-source fusion ===")
m3 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.2,
        frozenset({"c"}): 0.3,
        frozenset({"a", "b", "c"}): 0.5,
    },
)
print("Source 3:")
print(table(m3))
print()

for rule in ["dempster", "yager", "dubois-prade", "pcr6", "murphy"]:
    try:
        result = combine_multiple([m1, m2, m3], rule=rule)
        print(f"  {rule:15s} → focal elements: {len(result)}")
    except Exception as e:
        print(f"  {rule:15s} → error: {e}")
print()

# ── 9. Comparison on a high-conflict scenario ────────────────────────

print("=" * 60)
print("=== High-conflict scenario ===")
# Sensor 1 says "a" with high confidence
s1 = MassFunction(frame, named_focal_elements={
    frozenset({"a"}): 0.9,
    frozenset({"a", "b", "c"}): 0.1,
})
# Sensor 2 says "b" with high confidence (contradicts sensor 1)
s2 = MassFunction(frame, named_focal_elements={
    frozenset({"b"}): 0.9,
    frozenset({"a", "b", "c"}): 0.1,
})
print(f"  Conflict K = {s1.conflict(s2):.4f}")
print()

rules = {
    "Dempster": s1.combine_dempster(s2),
    "Yager": s1.combine_yager(s2),
    "Dubois-Prade": s1.combine_dubois_prade(s2),
    "PCR6": s1.combine_pcr6(s2),
    "Murphy": combine_murphy(s1, s2),
}

for name, result in rules.items():
    bp = result.pignistic()
    top = max(bp, key=bp.get)  # type: ignore[arg-type]
    print(f"  {name:15s} → BetP(a)={bp['a']:.4f}  BetP(b)={bp['b']:.4f}  "
          f"BetP(c)={bp['c']:.4f}  [best: {top}]")
