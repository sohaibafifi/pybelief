"""Basic usage of pybelief — Dempster-Shafer belief functions."""

from pybelief import MassFunction

# ── 1. Create mass functions ──────────────────────────────────────────────

frame = ["a", "b", "c"]

# Using label sets (human-friendly)
m1 = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.3,
        frozenset({"a", "b"}): 0.2,
        frozenset({"a", "b", "c"}): 0.5,
    },
)
print("m1:", m1)
print()

# Using bitmasks (compact — bit i = element i)
# {a}=0b001=1, {a,b}=0b011=3, {a,b,c}=0b111=7
m2 = MassFunction(frame, {0b010: 0.5, 0b011: 0.2, 0b111: 0.3})
print("m2:", m2)
print()

# ── 2. Belief, plausibility, commonality ──────────────────────────────────

print("=== Transforms for m1 ===")
print(f"  Bel({{a}})     = {m1.belief({'a'}):.4f}")
print(f"  Bel({{a,b}})   = {m1.belief({'a', 'b'}):.4f}")
print(f"  Pl({{a}})      = {m1.plausibility({'a'}):.4f}")
print(f"  Pl({{b}})      = {m1.plausibility({'b'}):.4f}")
print(f"  Q({{a}})       = {m1.commonality({'a'}):.4f}")
print(f"  Q({{a,b}})     = {m1.commonality({'a', 'b'}):.4f}")
print()

# ── 3. Pignistic probability ─────────────────────────────────────────────

bp = m1.pignistic()
print("Pignistic transform of m1:")
for label, prob in bp.items():
    print(f"  BetP({label}) = {prob:.4f}")
print()

# ── 4. Combination rules ─────────────────────────────────────────────────

# Dempster's rule (normalized)
m12_dempster = m1 & m2  # shorthand for m1.combine_dempster(m2)
print("Dempster combination (m1 & m2):")
print(m12_dempster)
print()

# Conjunctive / TBM (unnormalized — mass on empty set = conflict)
m12_conj = m1 | m2  # shorthand for m1.combine_conjunctive(m2)
print("Conjunctive combination (m1 | m2):")
print(m12_conj)
print(f"  Conflict mass on empty set: {m12_conj[0]:.4f}")
print()

# Yager's rule (conflict mass → full frame)
m12_yager = m1.combine_yager(m2)
print("Yager combination:")
print(m12_yager)
print()

# ── 5. Discounting ───────────────────────────────────────────────────────

m1_discounted = m1.discount(0.3)  # 30% unreliable
print("m1 discounted at alpha=0.3:")
print(m1_discounted)
print()

# ── 6. Conditioning ──────────────────────────────────────────────────────

m1_cond = m1.condition({"a", "b"})
print("m1 conditioned on {a, b}:")
print(m1_cond)
print()

# ── 7. Factory methods ───────────────────────────────────────────────────

vacuous = MassFunction.vacuous(frame)
print("Vacuous (total ignorance):", vacuous)

certain = MassFunction.certain(frame, "b")
print("Certain on b:", certain)

bayesian = MassFunction.from_bayesian(frame, {"a": 0.5, "b": 0.3, "c": 0.2})
print("Bayesian:", bayesian)
print()

# ── 8. Information measures ──────────────────────────────────────────────

print(f"Non-specificity of m1:  {m1.specificity():.4f}")
print(f"Deng entropy of m1:     {m1.entropy_deng():.4f}")
print(f"Conflict(m1, m2):       {m1.conflict(m2):.4f}")
