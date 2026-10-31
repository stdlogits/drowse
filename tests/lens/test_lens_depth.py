"""The depth rule: S below the crossover band, R above it, linear inside."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from drowse.lens.depth import LensBand, kind_weights, layer_depth


def test_default_band_is_15_to_35_percent_of_depth() -> None:
    band = LensBand()
    assert (band.lo, band.hi) == (0.15, 0.35)
    weights = kind_weights(21)  # depth = layer / 20
    assert weights[0] == (1.0, 0.0)
    assert weights[3] == (1.0, 0.0)            # depth 0.15: still all S
    assert weights[4] == pytest.approx((0.75, 0.25))  # depth 0.20
    assert weights[5] == pytest.approx((0.5, 0.5))    # depth 0.25: the middle
    assert weights[7] == (0.0, 1.0)            # depth 0.35: all R
    assert weights[20] == (0.0, 1.0)


@pytest.mark.parametrize("n_layers", [1, 2, 5, 36, 48])
def test_weights_sum_to_one_and_s_never_rises_with_depth(n_layers: int) -> None:
    weights = kind_weights(n_layers, LensBand(0.2, 0.6))
    assert len(weights) == n_layers
    assert all(w_s + w_r == pytest.approx(1.0) for w_s, w_r in weights)
    assert all(0.0 <= w_s <= 1.0 for w_s, _ in weights)
    s = [w_s for w_s, _ in weights]
    assert s == sorted(s, reverse=True)


def test_a_degenerate_band_is_a_hard_switch() -> None:
    weights = kind_weights(11, LensBand(0.5, 0.5))
    assert [w_s for w_s, _ in weights] == [1.0] * 6 + [0.0] * 5


def test_depth_is_the_readout_depth() -> None:
    assert layer_depth(0, 36) == 0.0 and layer_depth(35, 36) == 1.0
    assert layer_depth(0, 1) == 0.0
    assert kind_weights(1) == [(1.0, 0.0)]


@pytest.mark.parametrize("lo,hi", [(-0.1, 0.3), (0.4, 0.3), (0.2, 1.2)])
def test_invalid_bands_are_refused(lo: float, hi: float) -> None:
    with pytest.raises(ValueError, match="lens band"):
        LensBand(lo, hi)


def test_zero_layers_are_refused() -> None:
    with pytest.raises(ValueError):
        kind_weights(0)


def test_the_module_is_torch_free() -> None:
    import drowse.lens.depth as depth

    tree = ast.parse(Path(depth.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert "torch" not in imported
