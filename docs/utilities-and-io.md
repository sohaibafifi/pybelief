# Utilities and I/O

This document covers the formatting, export, and credal-set helpers exposed by
`pybelief`.

## Pretty printing

- `table(m, columns=("m", "Bel", "Pl", "Q"), sort_by="cardinality", empty_set=False, precision=4)`

Features:

- configurable columns
- sorting by cardinality, mass, or mask
- optional empty-set row
- optional singleton `BetP` display

## Export helpers

- `to_csv()`
- `to_json()`
- `to_ibelief()`
- `from_ibelief()`
- `to_matlab()`

### JSON

`to_json(..., include_transforms=True)` exports:

- focal elements
- transform tables for `Bel`, `Pl`, and `Q`
- pignistic probabilities when defined

### R / MATLAB compatibility

- `to_ibelief()` and `from_ibelief()` use the bitmask-indexed mass vector
  convention
- `to_matlab()` exports a MATLAB-ready vector string

## Credal-set helpers


- `credal_set_constraints()`
- `credal_set_vertices()`

`credal_set_constraints()` returns singleton lower and upper probability
bounds:

- lower bound: `Bel({x})`
- upper bound: `Pl({x})`

`credal_set_vertices()` enumerates extreme points for small frames using a
permutation-based construction.

## Current limit

For frames larger than 10 elements, `credal_set_vertices()` raises
`ValueError`. 