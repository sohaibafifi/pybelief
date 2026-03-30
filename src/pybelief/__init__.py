"""Finite-frame Dempster-Shafer belief functions with sparse bitmask storage."""

from pybelief.mass import MassFunction
from pybelief.decision import DecisionProblem
from pybelief.combination import (
    combine_disjunctive,
    combine_dubois_prade,
    combine_pcr6,
    combine_murphy,
    combine_cautious,
    combine_bold,
    combine_multiple,
)
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
from pybelief.display import (
    table,
    to_csv,
    to_json,
    to_ibelief,
    from_ibelief,
    to_matlab,
    credal_set_constraints,
    credal_set_vertices,
)

__all__ = [
    "MassFunction",
    "DecisionProblem",
    "combine_disjunctive",
    "combine_dubois_prade",
    "combine_pcr6",
    "combine_murphy",
    "combine_cautious",
    "combine_bold",
    "combine_multiple",
    "jousselme",
    "euclidean",
    "bhattacharyya",
    "tessem",
    "conflict_distance",
    "cosine_similarity",
    "cosine_distance",
    "deng_relative_entropy",
    "inclusion_degree",
    "auto_conflict",
    "pignistic_l1",
    "pignistic_l2",
    "table",
    "to_csv",
    "to_json",
    "to_ibelief",
    "from_ibelief",
    "to_matlab",
    "credal_set_constraints",
    "credal_set_vertices",
]
__version__ = "0.1.0"
