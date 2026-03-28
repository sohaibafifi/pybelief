# Combination Rules

The `combination` module provides fusion operators for combining evidence
from multiple sources, beyond the basic rules already on `MassFunction`.

## Quick reference

| Method / Function | Description |
|---|---|
| `m1.combine_dempster(m2)` / `m1 & m2` | Dempster's rule (normalized conjunctive) |
| `m1.combine_conjunctive(m2)` / `m1 \| m2` | Conjunctive / TBM (unnormalized) |
| `m1.combine_yager(m2)` | Yager's rule (conflict → Ω) |
| `m1.combine_disjunctive(m2)` | Disjunctive rule (union-based) |
| `m1.combine_dubois_prade(m2)` | Dubois-Prade (conflict → union) |
| `m1.combine_pcr6(m2)` | PCR6 - proportional conflict redistribution |
| `m1.combine_cautious(m2)` | Cautious rule (Denœux) - idempotent, min in weight domain |
| `m1.combine_bold(m2)` | Bold rule (Denœux) - idempotent, max in weight domain |
| `combine_murphy(m1, m2, ...)` | Murphy's averaging-based combination |
| `combine_multiple([m1, m2, ...], rule)` | Multi-source fusion with any rule |

All pairwise rules are also available as standalone functions in
`pybelief.combination` (e.g. `combine_disjunctive(m1, m2)`).

## Rules in detail

### Disjunctive rule

The disjunctive rule uses set union instead of intersection:

    m₁₂(C) = Σ_{A∪B=C} m₁(A)·m₂(B)

Dual of the conjunctive rule. Appropriate when at least one source is
reliable but it is unknown which one. Always produces a valid (normalized)
mass function.

**Reference:** Ph. Smets, "The combination of evidence in the Transferable
Belief Model", *IEEE Trans. PAMI*, 12(5), pp. 447-458, 1990.

### Dubois-Prade

When two focal elements conflict (A∩B = ∅), the Dubois-Prade rule sends
the corresponding mass to A∪B instead of the empty set:

    m_DP(C) = Σ_{A∩B=C, C≠∅} m₁(A)·m₂(B)
            + Σ_{A∪B=C, A∩B=∅} m₁(A)·m₂(B)

Unlike Dempster's rule, no mass is lost or renormalized. Unlike Yager's
rule, conflict mass is assigned to the specific union (not the full frame).

**Reference:** D. Dubois and H. Prade, "Representation and combination of
uncertainty with belief functions and possibility measures",
*Computational Intelligence*, 4(3), pp. 244-264, 1988.

### PCR6 (Proportional Conflict Redistribution #6)

The PCR6 rule starts with the conjunctive result and then redistributes
each conflict term m₁(A)·m₂(B) (where A∩B=∅) back to A and B in
proportion to each source's commitment:

    to A: m₁(A)²·m₂(B) / (m₁(A) + m₂(B))
    to B: m₂(B)²·m₁(A) / (m₁(A) + m₂(B))

Preserves all information; no renormalization needed.

**Reference:** F. Smarandache and J. Dezert, "Proportional Conflict
Redistribution Rules for Information Fusion", in *Advances and Applications
of DSmT for Information Fusion*, vol. 2, ch. 1, 2006.

### Murphy's average combination

A heuristic that mitigates the effect of conflicting sources:

1. Compute the element-wise average: m_avg(A) = (1/n) Σᵢ mᵢ(A)
2. Apply Dempster's rule (n-1) times on `m_avg` with itself

**Reference:** C.K. Murphy, "Combining belief functions when evidence
conflicts", *Decision Support Systems*, 29(1), pp. 1-9, 2000.

### Cautious rule (Denœux)

An idempotent conjunctive rule based on the canonical decomposition via
weight functions. The combination takes the element-wise *minimum* in
the weight-function domain:

    w₁₂(A) = min(w₁(A), w₂(A))  for all ∅ ⊂ A ⊂ Ω

The weight function w is obtained from the commonality function q via
Möbius inversion in the log domain (the canonical decomposition of Smets).

Key property: **idempotent** - m ⊕_cautious m = m.

Requires non-dogmatic mass functions (all commonality values strictly
positive).

**References:**
- T. Denœux, "Conjunctive and disjunctive combination of belief functions
  induced by nondistinct bodies of evidence", *Artificial Intelligence*,
  172(2-3), pp. 234-264, 2008.
- Ph. Smets, "The canonical decomposition of a weighted belief", in
  *Proc. IJCAI*, pp. 1896-1901, 1995.

### Bold rule (Denœux)

The disjunctive counterpart of the cautious rule. Takes the element-wise
*maximum* in the disjunctive weight-function domain:

    v₁₂(A) = max(v₁(A), v₂(A))  for all ∅ ⊂ A ⊂ Ω

Also **idempotent**.

Requires that Bel(A) > 0 for all proper non-empty subsets (non-vacuous
in the disjunctive sense).

**Reference:** T. Denœux, "Conjunctive and disjunctive combination of
belief functions induced by nondistinct bodies of evidence", *Artificial
Intelligence*, 172(2-3), pp. 234-264, 2008.

### Multi-source fusion

`combine_multiple(masses, rule)` applies any rule to a list of mass
functions:

```python
from pybelief import combine_multiple

result = combine_multiple([m1, m2, m3], rule="dempster")
```

Supported rules: `"dempster"`, `"conjunctive"`, `"disjunctive"`,
`"yager"`, `"dubois-prade"`, `"pcr6"`, `"murphy"`, `"cautious"`,
`"bold"`.

For sequential rules the combination is applied pairwise left to right.
Murphy's rule uses its own averaging procedure.

## Comparison of rules

| Rule | Normalized | Commutative | Associative | Idempotent | Handles high conflict well |
|---|---|---|---|---|---|
| Dempster | yes | yes | yes | no | no |
| Conjunctive | no | yes | yes | no | N/A |
| Disjunctive | yes | yes | yes | no | N/A |
| Dubois-Prade | yes | yes | no | no | yes |
| PCR6 | yes | yes | no | no | yes |
| Murphy | yes | - | - | no | yes |
| Cautious | yes | yes | yes | **yes** | yes |
| Bold | yes | yes | yes | **yes** | yes |
