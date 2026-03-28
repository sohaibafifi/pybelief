# Mass Function

`MassFunction` is the core data structure of the library. It represents a
finite-frame Dempster-Shafer basic belief assignment using sparse bitmask
storage.



### Construction

| Method | Description |
|--------|-------------|
| `MassFunction(frame, focal_elements={mask: mass})` | From bitmask dict |
| `MassFunction(frame, named_focal_elements={labels: mass})` | From label-set dict |
| `MassFunction.vacuous(frame)` | Total ignorance |
| `MassFunction.certain(frame, element)` | All mass on one element |
| `MassFunction.from_bayesian(frame, {label: prob})` | Probability distribution |

### Transforms

| Method | Description |
|--------|-------------|
| `m.belief(A)` | Bel(A) - sum of masses of subsets of A |
| `m.plausibility(A)` | Pl(A) - sum of masses intersecting A |
| `m.commonality(A)` | Q(A) - sum of masses of supersets of A |
| `m.belief_function()` | Bel for all 2^n subsets (fast zeta) |
| `m.plausibility_function()` | Pl for all 2^n subsets |
| `m.commonality_function()` | Q for all 2^n subsets |
| `m.pignistic()` | BetP probability transform |
| `m.plausibility_transform()` | Normalized singleton plausibilities |

### Operations

| Method | Description |
|--------|-------------|
| `m1 & m2` / `m1.combine_dempster(m2)` | Dempster's rule (normalized) |
| `m1 \| m2` / `m1.combine_conjunctive(m2)` | Conjunctive / TBM (unnormalized) |
| `m1.combine_yager(m2)` | Yager's rule |
| `m.discount(alpha)` | Shafer's discounting |
| `m.condition(event)` | Dempster conditioning |
| `m1.conflict(m2)` | Conflict mass K |

### Measures

| Method | Description |
|--------|-------------|
| `m.specificity()` | Non-specificity N(m) |
| `m.entropy_deng()` | Deng entropy |


## Serialization helpers

- `to_dict()`
- `from_dict()`
- `mask_to_set()`
- `set_to_mask()`

## Intended use

The class supports both normalized belief functions and unnormalized
conjunctive masses produced by TBM-style fusion. Decision criteria and credal
set utilities should be interpreted primarily in the normalized
Dempster-Shafer setting.
