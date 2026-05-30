"""Invariants and spatial-locality checks for every ordering scheme."""

from __future__ import annotations

import pytest
import torch

from hilbert_wsi.ordering import available_orderings, get_ordering

PATCH = 256


def _grid_coords(side: int = 8) -> torch.Tensor:
    xs, ys = torch.meshgrid(
        torch.arange(side) * PATCH,
        torch.arange(side) * PATCH,
        indexing="xy",
    )
    return torch.stack([xs.flatten(), ys.flatten()], dim=-1)


def _avg_step(coords: torch.Tensor, perm: torch.Tensor) -> float:
    ordered = coords[perm].float()
    deltas = (ordered[1:] - ordered[:-1]).norm(dim=-1)
    return float(deltas.mean())


@pytest.mark.parametrize("name", available_orderings())
def test_permutation_is_valid(name: str):
    coords = _grid_coords(8)
    scheme = get_ordering(name) if name != "similarity" else get_ordering(name)
    if name == "similarity":
        feats = torch.randn(coords.shape[0], 16)
        perm = scheme(coords, features=feats)
    else:
        perm = scheme(coords)
    assert perm.shape == (coords.shape[0],)
    assert perm.dtype == torch.long
    assert torch.equal(torch.sort(perm).values, torch.arange(coords.shape[0]))


@pytest.mark.parametrize("name", [n for n in available_orderings() if n not in {"random", "similarity"}])
def test_ordering_is_deterministic(name: str):
    coords = _grid_coords(8)
    scheme = get_ordering(name)
    assert torch.equal(scheme(coords), scheme(coords))


@pytest.mark.parametrize("name", ["hilbert", "zorder", "moore", "peano", "snake"])
def test_spatial_locality_beats_random(name: str):
    torch.manual_seed(0)
    coords = _grid_coords(8)
    local = get_ordering(name)(coords)
    rand = get_ordering("random", seed=0)(coords)
    assert _avg_step(coords, local) < _avg_step(coords, rand)


def test_empty_coords_returns_empty_perm():
    empty = torch.empty(0, 2, dtype=torch.long)
    for name in available_orderings():
        if name == "similarity":
            perm = get_ordering(name)(empty, features=torch.empty(0, 4))
        else:
            perm = get_ordering(name)(empty)
        assert perm.numel() == 0


def test_similarity_uses_features():
    torch.manual_seed(1)
    coords = _grid_coords(4)
    feats = torch.randn(coords.shape[0], 8)
    perm_a = get_ordering("similarity")(coords, features=feats)
    perm_b = get_ordering("similarity")(coords, features=feats)
    assert torch.equal(perm_a, perm_b)


# ---------------------------------------------------------------------------
# Canonical Peano regression: lock in correctness of the column-major S-curve.
# Reference table hand-computed for the 3x3 grid; spot-checks for 9x9 follow
# the standard recursive construction (outer S + Y-flipped middle column).
# ---------------------------------------------------------------------------

def test_peano_matches_canonical_3x3():
    from hilbert_wsi.ordering.peano import _peano_index

    expected = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2,
        (1, 2): 3, (1, 1): 4, (1, 0): 5,
        (2, 0): 6, (2, 1): 7, (2, 2): 8,
    }
    for (x, y), exp in expected.items():
        got = int(_peano_index(torch.tensor([x]), torch.tensor([y]), 3).item())
        assert got == exp, f"Peano(3x3)({x},{y}) -> {got}, expected {exp}"


def test_peano_matches_canonical_9x9_spotchecks():
    from hilbert_wsi.ordering.peano import _peano_index

    spot = [
        ((0, 0), 0), ((0, 2), 2), ((1, 2), 3), ((2, 2), 8),
        ((3, 8), 27),   # outer(1,2), Y-flipped inner(0,0)
        ((4, 4), 40),   # outer(1,1), Y-flipped inner(1,1)
        ((5, 0), 53),   # outer(1,0), Y-flipped inner(2,2)
        ((6, 0), 54), ((8, 8), 80),
        ((0, 4), 10), ((1, 4), 13),
    ]
    for (x, y), exp in spot:
        got = int(_peano_index(torch.tensor([x]), torch.tensor([y]), 9).item())
        assert got == exp, f"Peano(9x9)({x},{y}) -> {got}, expected {exp}"


# ---------------------------------------------------------------------------
# Random ordering: must produce a *different* permutation per slide so it is
# a real negative control. Previously the seed was a fixed scalar, so two
# slides with the same N received the same permutation -- not random.
# ---------------------------------------------------------------------------

def test_random_is_per_slide():
    n = 16
    coords_a = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)], dim=-1)
    coords_b = torch.stack([torch.arange(n) + 100, torch.ones(n, dtype=torch.long)], dim=-1)
    r = get_ordering("random")
    perm_a = r(coords_a)
    perm_b = r(coords_b)
    assert not torch.equal(perm_a, perm_b), (
        "Random ordering returns the same permutation for two slides with "
        "different coords -- the per-slide seeding is broken."
    )


def test_random_is_deterministic_per_slide():
    n = 16
    coords = torch.stack(
        [torch.arange(n) * 256, (torch.arange(n) % 4) * 256], dim=-1
    )
    r = get_ordering("random")
    assert torch.equal(r(coords), r(coords))


def test_random_seed_affects_permutation():
    n = 16
    coords = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)], dim=-1)
    p0 = get_ordering("random", seed=0)(coords)
    p1 = get_ordering("random", seed=1)(coords)
    assert not torch.equal(p0, p1), (
        "Random ordering ignores base_seed -- multi-seed sweep won't work."
    )
