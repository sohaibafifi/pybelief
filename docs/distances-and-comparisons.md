# Distances and Comparisons

The `distances` module provides 12 measures for comparing mass functions:
metric distances, divergences, similarity measures, and internal consistency
checks.

## Quick reference

| Function | Type | Symmetric | Range |
|---|---|---|---|
| `jousselme(m1, m2)` | metric | yes | [0, 1] |
| `euclidean(m1, m2)` | metric | yes | [0, +inf) |
| `bhattacharyya(m1, m2)` | distance (on BetP) | yes | [0, +inf] |
| `tessem(m1, m2)` | distance (L-inf on BetP) | yes | [0, 1] |
| `conflict_distance(m1, m2)` | distance | yes | [0, +inf] |
| `cosine_similarity(m1, m2)` | similarity | yes | [0, 1] |
| `cosine_distance(m1, m2)` | distance | yes | [0, 1] |
| `deng_relative_entropy(m1, m2)` | divergence | **no** | [0, +inf] |
| `inclusion_degree(m1, m2)` | similarity | **no** | [0, 1] |
| `auto_conflict(m)` | self-measure | - | [0, 1] |
| `pignistic_l1(m1, m2)` | distance (L1 on BetP) | yes | [0, 2] |
| `pignistic_l2(m1, m2)` | distance (L2 on BetP) | yes | [0, sqrt(2)] |

All pairwise functions require mass functions on the same frame.

## Distances on mass vectors

### Jousselme distance

The standard distance in the Dempster-Shafer community. Uses the Jaccard
similarity matrix D(A,B) = |A inter B| / |A union B|:

    d_J(m1, m2) = sqrt(1/2 (m1 - m2)^T D (m1 - m2))

Properties: true metric, bounded in [0, 1], accounts for set-theoretic
overlap between focal elements.

**Reference:** Jousselme, A.-L., Grenier, D., & Bosse, E. (2001). A new
distance between two bodies of evidence. *Information Fusion*, 2(2), 91-101.

### Euclidean distance

Simple L2 norm on the mass vectors:

    d_E(m1, m2) = sqrt(sum_A (m1(A) - m2(A))^2)

Fast and sparse-friendly. Does not account for set overlap.

**Reference:** Jousselme, A.-L. & Maupin, P. (2012). Distances in evidence
theory: Comprehensive survey and generalizations. *Int. J. Approximate
Reasoning*, 53(2), 118-145.

### Cosine similarity / distance

Cosine of the angle between mass vectors:

    sim(m1, m2) = sum_A m1(A) m2(A) / (||m1|| ||m2||)
    dist = 1 - sim

**Reference:** Jousselme, A.-L. & Maupin, P. (2012). *Int. J. Approximate
Reasoning*, 53(2), 118-145.

## Distances on pignistic probabilities

These measures first compute the pignistic transform BetP, then compare the
resulting probability distributions.

**TBM edge case:** The pignistic transform is undefined when all mass sits
on the empty set (a degenerate state reachable via conjunctive combination
of totally conflicting sources). All four pignistic-based functions
(`bhattacharyya`, `tessem`, `pignistic_l1`, `pignistic_l2`) raise
`ValueError` in this case. Mass-vector distances (`jousselme`, `euclidean`,
`cosine_distance`) work on any mass function including unnormalized TBM
masses.

### Tessem distance

L-infinity (max absolute difference) on pignistic probabilities:

    d_T(m1, m2) = max_x |BetP1(x) - BetP2(x)|

**Reference:** Tessem, B. (1993). Approximations for efficient computation
in the theory of evidence. *Artificial Intelligence*, 61(2), 315-329.

### Pignistic L1 and L2

Manhattan and Euclidean distances on pignistic probabilities:

    d_L1 = sum_x |BetP1(x) - BetP2(x)|
    d_L2 = sqrt(sum_x (BetP1(x) - BetP2(x))^2)

**Reference:** Tessem, B. (1993). *Artificial Intelligence*, 61(2), 315-329.

### Bhattacharyya distance

    d_B = -ln(sum_x sqrt(BetP1(x) BetP2(x)))

Returns `inf` when the supports are disjoint (Bhattacharyya coefficient = 0).

**References:**
- Bhattacharyya, A. (1943). On a measure of divergence between two
  statistical populations. *Bull. Calcutta Math. Soc.*, 35, 99-109.
- Zouhal, L.M. & Denoeux, T. (1998). An evidence-theoretic k-NN rule.
  *IEEE Trans. SMC-C*, 28(2), 263-271.

## Conflict-based measures

### Conflict distance

Weight of evidence derived from the conflict mass K:

    d_K(m1, m2) = -log2(1 - K)

where K = sum_{A inter B = empty} m1(A) m2(B). Returns `inf` for total
conflict (K = 1).

**References:**
- Smets, P. (1990). The combination of evidence in the TBM. *IEEE Trans.
  PAMI*, 12(5), 447-458.
- Ristic, B. & Smets, P. (2008). Global cost of a decision in the TBM.
  *Int. J. Approximate Reasoning*, 48(2), 327-341.

### Auto-conflict

Conflict of a mass function with itself:

    K_auto(m) = sum_{A inter B = empty} m(A) m(B)

Measures internal incoherence. A single focal element always has zero
auto-conflict. Mass functions split across disjoint singletons have high
auto-conflict.

**Reference:** Daniel, M. (2006). Conflicts within and between belief
functions. *IPMU 2006*, pp. 696-703.

## Divergences (asymmetric)

### Deng relative entropy

Generalized KL divergence for mass functions:

    D(m1 || m2) = sum_A m1(A) ln(m1(A) / m2(A))

Returns `inf` if any focal element of m1 has zero mass in m2. For Bayesian
masses, this equals the standard KL divergence. **Not symmetric.**

**Reference:** Deng, Y. (2016). Deng entropy. *Chaos, Solitons & Fractals*,
91, 549-553.

### Inclusion degree

Measures how much the evidence in m1 is "included" in m2:

    Inc(m1, m2) = sum_A sum_B m1(A) m2(B) |A inter B| / |A|

Returns 1.0 when m2 is vacuous (total ignorance includes everything).
**Not symmetric.**

**Reference:** Zouhal, L.M. & Denoeux, T. (1998). An evidence-theoretic
k-NN rule. *IEEE Trans. SMC-C*, 28(2), 263-271.

## Choosing a distance

| Use case | Recommended |
|---|---|
| General-purpose comparison | `jousselme` |
| Fast approximation | `euclidean` |
| Decision-oriented comparison | `tessem` or `pignistic_l1` |
| Clustering mass functions | `jousselme` or `cosine_distance` |
| Detecting conflicting sources | `conflict_distance` or `auto_conflict` |
| Asymmetric divergence (e.g. fitting) | `deng_relative_entropy` |
| Checking if evidence is subsumed | `inclusion_degree` |

## General survey reference

Jousselme, A.-L. & Maupin, P. (2012). Distances in evidence theory:
Comprehensive survey and generalizations. *Int. J. Approximate Reasoning*,
53(2), 118-145.
