# Decision Making

The decision module implements standard action-selection criteria for belief
functions on a finite state space.

## Main class

```python
from pybelief import DecisionProblem
```

```python
DecisionProblem(mass, utilities)
```

where:

- `mass` is a `MassFunction` on the states of nature
- `utilities` is `{action: {state: utility}}`

Every action must provide one utility value for each state in the frame.

## Lower and upper expected utility


- `lower_expectation(action)`
- `upper_expectation(action)`
- `expectation_interval(action)`

The code uses the Choquet formulas:

- lower expectation:
  `sum_A m(A) * min_{theta in A} U(action, theta)`
- upper expectation:
  `sum_A m(A) * max_{theta in A} U(action, theta)`

For Bayesian masses, lower and upper expectations coincide with ordinary
expected utility.

## Decision criteria

- `maximin()`
- `maximax()`
- `hurwicz(alpha=0.5)`
- `interval_dominance()`
- `maximality()`
- `e_admissibility()`
- `summary()`

## Inclusion hierarchy

For normalized belief functions, the criteria satisfy:

`E-admissibility ⊆ maximality ⊆ interval dominance`

This is the hierarchy implemented and numerically checked against random test instances.
TODO: Add a reference to the literature on this hierarchy.

## E-admissibility

`e_admissibility()` is implemented as a feasibility LP:

- variables are singleton probabilities `p(state)`
- constraints encode the credal set
  `Bel(A) <= P(A) <= Pl(A)` for all proper non-empty subsets
- action-optimality constraints enforce that the tested action is at least as
  good as every competitor under `P`

Dependency:

- requires `scipy.optimize.linprog`

If SciPy is not installed, the method raises `ImportError`.

## Scope

The mathematical interpretation of this module is the normalized
Dempster-Shafer setting. While the class accepts any `MassFunction`, decision
criteria are documented and validated for normalized masses.
