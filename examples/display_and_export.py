"""Display and export utilities for pybelief mass functions."""

from pybelief import (
    MassFunction,
    table,
    to_csv,
    to_json,
    to_ibelief,
    to_matlab,
    credal_set_constraints,
    credal_set_vertices,
)

# ── Create a mass function ───────────────────────────────────────────

frame = ["a", "b", "c"]
m = MassFunction(
    frame,
    named_focal_elements={
        frozenset({"a"}): 0.3,
        frozenset({"a", "b"}): 0.2,
        frozenset({"a", "b", "c"}): 0.5,
    },
)

# ── 1. Pretty-print table ───────────────────────────────────────────

print("=== Default table (m, Bel, Pl, Q) ===")
print(table(m))
print()

print("=== With pignistic column ===")
print(table(m, columns=("m", "Bel", "Pl", "BetP")))
print()

print("=== Sorted by mass (descending) ===")
print(table(m, sort_by="mass"))
print()

print("=== Only focal elements (mass > 0) with higher precision ===")
print(table(m, columns=("m",), precision=6))
print()

print("=== Including the empty set row ===")
print(table(m, empty_set=True))
print()

# ── 2. CSV export ────────────────────────────────────────────────────

print("=== CSV output ===")
print(to_csv(m))

# ── 3. JSON export ───────────────────────────────────────────────────

print("=== JSON (basic) ===")
print(to_json(m))
print()

print("=== JSON (with all transforms) ===")
print(to_json(m, include_transforms=True))
print()

# ── 4. R ibelief / MATLAB compatibility ─────────────────────────────

print("=== R ibelief vector ===")
vec = to_ibelief(m)
print(f"  {vec}")
print()

print("=== MATLAB BFT string ===")
print(f"  {to_matlab(m)}")
print()

# ── 5. Credal set ───────────────────────────────────────────────────

print("=== Credal set bounds (singleton probabilities) ===")
labels, bounds = credal_set_constraints(m)
for label, (lo, hi) in zip(labels, bounds):
    print(f"  P({label}) in [{lo:.4f}, {hi:.4f}]")
print()

print("=== Credal set vertices ===")
vertices = credal_set_vertices(m)
print(f"  Found {len(vertices)} extreme point(s):")
for i, v in enumerate(vertices):
    probs = ", ".join(f"P({k})={val:.4f}" for k, val in v.items())
    print(f"    vertex {i+1}: {probs}")
print()

# ── 6. Roundtrip: ibelief → MassFunction ────────────────────────────

from pybelief import from_ibelief

m_restored = from_ibelief(frame, vec)
print("=== Roundtrip: to_ibelief → from_ibelief ===")
print(f"  Original:  {m}")
print(f"  Restored:  {m_restored}")
print(f"  Equal: {m == m_restored}")
