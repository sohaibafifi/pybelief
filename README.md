# pybelief

Finite-frame Dempster-Shafer belief functions with sparse bitmask storage.

## Installation

```bash
pip install pybelief
```

## Quick start

```python
from pybelief import DecisionProblem, MassFunction

frame = ["a", "b", "c"]

# Create mass functions using label sets
m1 = MassFunction(frame, named_focal_elements={
    frozenset({"a"}): 0.3,
    frozenset({"a", "b"}): 0.2,
    frozenset({"a", "b", "c"}): 0.5,
})

m2 = MassFunction(frame, named_focal_elements={
    frozenset({"b"}): 0.5,
    frozenset({"a", "b"}): 0.2,
    frozenset({"a", "b", "c"}): 0.3,
})

# Query belief, plausibility, commonality
m1.belief({"a"})          # 0.3
m1.plausibility({"b"})    # 0.7
m1.commonality({"a"})     # 1.0

# Combine sources
m12 = m1 & m2                    # Dempster's rule
m12 = m1 | m2                    # Conjunctive (TBM)
m12 = m1.combine_yager(m2)       # Yager's rule

# Decision making
dp = DecisionProblem(m1, {
    "accept": {"a": 10, "b": 4, "c": 1},
    "reject": {"a": 0, "b": 3, "c": 6},
})
dp.hurwicz(alpha=0.5)
```

## API overview

- [Mass function core](docs/mass-function.md)
- [Decision making](docs/decision-making.md)
- [Utilities and I/O](docs/utilities-and-io.md)

## Bitmask convention

Element *i* of the frame maps to bit *i*. For `frame = ["a", "b", "c"]`:

| Set       | Binary  | Int |
|-----------|---------|-----|
| {a}       | `001`   | 1   |
| {b}       | `010`   | 2   |
| {a, b}    | `011`   | 3   |
| {c}       | `100`   | 4   |
| {a, c}    | `101`   | 5   |
| {b, c}    | `110`   | 6   |
| {a, b, c} | `111`   | 7   |

You can use either bitmasks or label sets in all API calls:

```python
m.belief(0b011)          # bitmask
m.belief({"a", "b"})     # label set — same result
```

## Performance

Focal elements are stored sparsely. Combination runs in O(|F1| x |F2|) — fast when mass functions are sparse. Full transforms use the fast zeta transform in O(n * 2^n). Practical for frames up to ~20 elements.

## Citation

If you use pybelief in your research, please cite:

```bibtex
@software{afifi2025pybelief,
  author    = {Afifi, Sohaib},
  title     = {pybelief: Finite-Frame Dempster--Shafer Belief Functions in Python},
  year      = {2025},
  url       = {https://github.com/sohaibafifi/pybelief},
}
```

## License

MIT
