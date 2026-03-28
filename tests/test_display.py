"""Tests for pybelief.display - pretty-printing, I/O, credal set."""

import json
import pytest
from pybelief import (
    MassFunction,
    table,
    to_csv,
    to_json,
    to_ibelief,
    from_ibelief,
    to_matlab,
    credal_set_constraints,
    credal_set_vertices,
)


FRAME = ["a", "b", "c"]


@pytest.fixture()
def m():
    return MassFunction(FRAME, {0b001: 0.3, 0b011: 0.2, 0b111: 0.5})


# ── table ─────────────────────────────────────────────────────────────


class TestTable:
    def test_basic(self, m):
        t = table(m)
        assert "Bel" in t
        assert "Pl" in t
        assert "{a}" in t
        # Should have header + separator + 7 rows (all non-empty subsets)
        lines = t.strip().split("\n")
        assert len(lines) == 9  # header + sep + 7 subsets

    def test_columns(self, m):
        t = table(m, columns=("m",))
        assert "Bel" not in t
        assert "m" in t.split("\n")[0]

    def test_sort_by_mass(self, m):
        t = table(m, sort_by="mass")
        lines = t.strip().split("\n")
        # First data row should be the largest mass
        assert "0.5000" in lines[2]

    def test_empty_set(self, m):
        t = table(m, empty_set=True)
        assert "{}" in t

    def test_betp_column(self, m):
        t = table(m, columns=("m", "BetP"))
        assert "BetP" in t
        # Singletons should have BetP values
        assert "-" in t  # non-singletons show "-"

    def test_precision(self, m):
        t = table(m, precision=2)
        assert "0.30" in t


# ── CSV ───────────────────────────────────────────────────────────────


class TestCSV:
    def test_basic(self, m):
        csv_str = to_csv(m)
        lines = csv_str.strip().split("\n")
        assert lines[0].startswith("Set")
        assert len(lines) == 8  # header + 7 subsets

    def test_columns(self, m):
        csv_str = to_csv(m, columns=("m", "Bel"))
        header = csv_str.split("\n")[0]
        assert "Pl" not in header
        assert "Bel" in header


# ── JSON ──────────────────────────────────────────────────────────────


class TestJSON:
    def test_basic(self, m):
        j = to_json(m)
        data = json.loads(j)
        assert "frame" in data
        assert "focal_elements" in data

    def test_with_transforms(self, m):
        j = to_json(m, include_transforms=True)
        data = json.loads(j)
        assert "transforms" in data
        assert "pignistic" in data

    def test_with_transforms_and_empty_set_mass(self):
        m = MassFunction(FRAME, {0: 0.2, 0b001: 0.3, 0b011: 0.2, 0b111: 0.3})
        j = to_json(m, include_transforms=True)
        data = json.loads(j)
        assert data["transforms"]["['a']"]["Pl"] == pytest.approx(0.8)
        assert data["transforms"]["['a', 'b', 'c']"]["Pl"] == pytest.approx(0.8)

    def test_roundtrip_via_json(self, m):
        j = to_json(m)
        data = json.loads(j)
        m2 = MassFunction.from_dict(data)
        assert m == m2


# ── ibelief compatibility ─────────────────────────────────────────────


class TestIbelief:
    def test_to_ibelief_length(self, m):
        vec = to_ibelief(m)
        assert len(vec) == 8  # 2^3

    def test_to_ibelief_values(self, m):
        vec = to_ibelief(m)
        assert vec[0b001] == pytest.approx(0.3)
        assert vec[0b011] == pytest.approx(0.2)
        assert vec[0b111] == pytest.approx(0.5)
        assert vec[0] == pytest.approx(0.0)

    def test_roundtrip(self, m):
        vec = to_ibelief(m)
        m2 = from_ibelief(FRAME, vec)
        assert m == m2

    def test_from_ibelief_bad_length(self):
        with pytest.raises(ValueError, match="does not match"):
            from_ibelief(["a", "b"], [0.0, 0.5, 0.5])


# ── MATLAB ────────────────────────────────────────────────────────────


class TestMatlab:
    def test_format(self, m):
        s = to_matlab(m)
        assert s.startswith("[")
        assert s.endswith("]")
        values = s[1:-1].split()
        assert len(values) == 8


# ── Credal set ────────────────────────────────────────────────────────


class TestCredalSet:
    def test_constraints(self, m):
        labels, bounds = credal_set_constraints(m)
        assert labels == FRAME
        assert len(bounds) == 3
        for lo, hi in bounds:
            assert lo <= hi + 1e-9
            assert lo >= 0 - 1e-9
            assert hi <= 1 + 1e-9

    def test_constraints_bayesian(self):
        m = MassFunction.from_bayesian(FRAME, {"a": 0.5, "b": 0.3, "c": 0.2})
        labels, bounds = credal_set_constraints(m)
        # For Bayesian, Bel({x}) = Pl({x}) = P(x)
        assert bounds[0][0] == pytest.approx(0.5)
        assert bounds[0][1] == pytest.approx(0.5)

    def test_vertices(self, m):
        verts = credal_set_vertices(m)
        assert len(verts) >= 1
        for v in verts:
            assert set(v.keys()) == set(FRAME)
            assert sum(v.values()) == pytest.approx(1.0)
            # Each vertex should satisfy Bel ≤ P ≤ Pl
            for lbl in FRAME:
                bel = m.belief({lbl})
                pl = m.plausibility({lbl})
                assert v[lbl] >= bel - 1e-9
                assert v[lbl] <= pl + 1e-9

    def test_vertices_bayesian(self):
        m = MassFunction.from_bayesian(FRAME, {"a": 0.5, "b": 0.3, "c": 0.2})
        verts = credal_set_vertices(m)
        # Bayesian → single vertex = the probability itself
        assert len(verts) == 1
        assert verts[0]["a"] == pytest.approx(0.5)

    def test_vertices_vacuous(self):
        m = MassFunction.vacuous(FRAME)
        verts = credal_set_vertices(m)
        # Vacuous → credal set = full simplex → n! vertices (permutations)
        # But many may coincide; at least n vertices
        assert len(verts) >= 3

    def test_vertices_too_large_frame(self):
        large_frame = [f"x{i}" for i in range(11)]
        m = MassFunction.vacuous(large_frame)
        with pytest.raises(ValueError, match="too large"):
            credal_set_vertices(m)
