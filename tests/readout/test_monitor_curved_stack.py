"""The curved probe's stacked read against the per-layer algorithm it batches.

A curved probe reads every layer at once over its ``_CurvedStack``: one
batched projection, one batched foot solve (``invert_stacked``), and the
share-weighted means accumulated as a left fold in layer order.  These tests
pin it to a straight per-layer implementation of the same read (per-layer
M-projection, ``invert_parameterization`` per layer, sequential weighted
sums), including layers of different rank.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from drowse.geometry.domains import BoxDomain
from drowse.geometry.invert import invert_parameterization, invert_stacked
from drowse.geometry.manifold import Manifold
from drowse.geometry.subspace import LayerSubspace, StackedRbf
from drowse.readout.monitor import Monitor, _mean_coordinate_space, _mean_domain_coordinates
from tests.support.curved import DIM, curved_stack_manifold as _curved
from tests.support.whitener import isotropic_whitener

def _monitor(m: Manifold) -> Monitor:
    mon = Monitor(whitener=isotropic_whitener(list(m.layers), DIM), n_layers=8)
    mon.add_probe("curve", m, top_n=3)
    return mon


def _reference(mon: Monitor, hidden: dict[int, torch.Tensor]) -> dict[str, Any]:
    """The per-layer read: projection, solve and weighted sums layer by layer."""
    probe = mon.attached_probes()["curve"]
    m = probe.manifold
    shared = [L for L in m.layers if L in hidden]
    total = sum(probe.share_weights[L] for L in shared)
    w = {L: probe.share_weights[L] / total for L in shared}
    whitener = mon.whitener
    frac_mean = dist = coords = mem = resid = None
    ref_layer = max(shared, key=lambda L: w[L])
    ref_coord = None
    for L in shared:
        wh, sub = probe.whitened[L], m.layers[L]
        h = hidden[L].to(torch.float32)
        x = h - wh.mean
        sx = whitener.apply_inv(L, h) - wh.s_mean
        g = wh.basis @ sx
        c = wh.m_r_inv @ g
        frac = (torch.sqrt((g * c).sum().clamp(min=0)) / torch.sqrt((x * sx).sum().clamp(min=0))).clamp(0, 1)
        d = torch.linalg.vector_norm(wh.node_white_aug - (c.reshape(1, -1) @ wh.chol), dim=-1) * w[L]
        pos, res = invert_parameterization(sub, m.domain, c.reshape(1, -1), m.node_coords)
        coord = pos.reshape(-1)
        if L == ref_layer:
            ref_coord = coord
        space = w[L] * _mean_coordinate_space(m.domain, coord)
        sig = sub.sigma_at(m.domain.embed(pos.reshape(1, -1))).reshape(1)
        mem_t = torch.exp(-(res.reshape(1) ** 2) / (2.0 * sig ** 2))
        r_t = res.reshape(1) / torch.linalg.vector_norm(c)
        frac_mean = w[L] * frac if frac_mean is None else frac_mean + w[L] * frac
        dist = d if dist is None else dist + d
        coords = space if coords is None else coords + space
        mem = w[L] * mem_t if mem is None else mem + w[L] * mem_t
        resid = w[L] * r_t if resid is None else resid + w[L] * r_t
    assert ref_coord is not None and coords is not None and dist is not None
    assert frac_mean is not None and mem is not None and resid is not None
    return {
        "fraction": float(frac_mean),
        "coords": _mean_domain_coordinates(m.domain, coords, ref_coord).tolist(),
        "membership": float(mem),
        "residual": float(resid),
        "nearest": torch.argsort(dist)[:3].tolist(),
        "distances": dist.tolist(),
    }


def _hidden(m: Manifold, gen: torch.Generator, layers: list[int] | None = None) -> dict[int, torch.Tensor]:
    out = {}
    for L, sub in m.layers.items():
        if layers is not None and L not in layers:
            continue
        coords = torch.rand((1, 1), generator=gen) * 1.6 - 0.8
        if isinstance(m.domain, BoxDomain) and m.domain.axes[0].periodic:
            coords = torch.rand((1, 1), generator=gen) * 6.0
        on = sub.eval_at(m.domain.embed(coords)).reshape(-1)
        out[L] = on + 0.05 * torch.randn(DIM, generator=gen)
    return out


@pytest.mark.parametrize("periodic", [False, True])
def test_stacked_read_matches_the_per_layer_read(periodic: bool) -> None:
    m = _curved(periodic=periodic)
    ranks = {sub.rank for sub in m.layers.values()}
    assert len(ranks) > 1, "the fixture must pad ranks"
    mon = _monitor(m)
    gen = torch.Generator().manual_seed(3)
    for _ in range(6):
        hidden = _hidden(m, gen)
        got = mon.score_single_token(hidden)["curve"]
        ref = _reference(mon, hidden)
        assert got.fraction == pytest.approx(ref["fraction"], abs=1e-5)
        assert list(got.coords) == pytest.approx(ref["coords"], abs=1e-4)
        assert got.membership == pytest.approx(ref["membership"], abs=1e-5)
        assert got.residual == pytest.approx(ref["residual"], abs=1e-5)
        # The curve passes through neutral at one node, so that node and neutral tie exactly and
        # their order is the platform's float rounding: the labels read must be the reference's
        # three nearest by distance, in order, whichever way a tie falls.
        labels = m.node_labels + ["neutral"]
        picked = [ref["distances"][labels.index(lab)] for lab, _d in got.nearest]
        assert picked == pytest.approx(sorted(ref["distances"])[:3], abs=1e-6)


def test_a_read_of_some_layers_reweighs_over_them() -> None:
    m = _curved(n_layers=5)
    mon = _monitor(m)
    gen = torch.Generator().manual_seed(5)
    hidden = _hidden(m, gen, layers=[0, 2, 3])
    got = mon.score_single_token(hidden)["curve"]
    ref = _reference(mon, hidden)
    assert set(got.fraction_per_layer) == {0, 2, 3}
    assert got.fraction == pytest.approx(ref["fraction"], abs=1e-5)
    assert list(got.coords) == pytest.approx(ref["coords"], abs=1e-4)
    assert got.membership == pytest.approx(ref["membership"], abs=1e-5)


def test_one_foot_solve_per_rank_whatever_the_layer_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import drowse.readout.monitor_attach as monitor_mod

    calls: list[int] = []

    def counting(surfaces: StackedRbf, *args: Any, **kwargs: Any) -> Any:
        calls.append(int(surfaces.node_params.shape[0]))
        return invert_stacked(surfaces, *args, **kwargs)

    monkeypatch.setattr(monitor_mod, "invert_stacked", counting)
    m = _curved(n_layers=6)
    mon = _monitor(m)
    mon.score_single_token(_hidden(m, torch.Generator().manual_seed(1)))
    ranks = [sub.rank for sub in m.layers.values()]
    assert sorted(calls) == sorted(ranks.count(r) for r in set(ranks))


def test_subspace_coords_drop_the_rank_padding() -> None:
    m = _curved()
    mon = _monitor(m)
    mon.set_subspace_coords(True)
    reading = mon.score_single_token(_hidden(m, torch.Generator().manual_seed(2)))["curve"]
    coords = reading.subspace_coords_per_layer
    assert coords is not None
    assert {L: len(v) for L, v in coords.items()} == {L: sub.rank for L, sub in m.layers.items()}


def test_invert_stacked_matches_invert_parameterization() -> None:
    m = _curved(periodic=True)
    gen = torch.Generator().manual_seed(4)
    by_rank: dict[int, list[LayerSubspace]] = {}
    for sub in m.layers.values():
        by_rank.setdefault(sub.rank, []).append(sub)
    assert len(by_rank) > 1
    for rank, subs in by_rank.items():
        surfaces = StackedRbf.build(subs, torch.device("cpu"))
        query = torch.randn(len(subs), rank, generator=gen)
        for warm in (None, torch.rand(len(subs), 1, generator=gen) * 6.0):
            pos, dist = invert_stacked(surfaces, m.domain, query, m.node_coords, warm_start=warm)
            for i, sub in enumerate(subs):
                p_ref, d_ref = invert_parameterization(
                    sub, m.domain, query[i:i + 1], m.node_coords,
                    warm_start=None if warm is None else warm[i:i + 1],
                )
                gap = (pos[i] - p_ref.reshape(-1) + 3.0) % 6.0 - 3.0
                assert float(gap.abs().max()) < 1e-4
                assert float(dist[i]) == pytest.approx(float(d_ref), abs=1e-5)


def test_a_surface_stack_holds_one_rank() -> None:
    subs = list(_curved().layers.values())
    with pytest.raises(ValueError, match="one rank"):
        StackedRbf.build(subs, torch.device("cpu"))


def test_cold_reads_neither_start_from_nor_move_the_warm_feet() -> None:
    """Regression: authored prompt tokens were scored on the decode loop's
    warm-started curved feet, so their rows disagreed with a replay's cold
    one-off read and moved the feet the next decoded token starts from.
    Inside ``cold_reads`` a read is the cold read, and the warm path resumes
    from where it was."""
    m = _curved(periodic=True)
    gen = torch.Generator().manual_seed(11)
    tokens = [_hidden(m, gen) for _ in range(4)]
    cold = _monitor(m)
    want = [cold.score_single_token(h)["curve"].coords for h in tokens]
    mon = _monitor(m)
    mon.enable_curved_warm(True)
    mon.score_single_token(tokens[0])
    feet = {key: foot.clone() for key, foot in mon._curved_feet.items()}
    with mon.cold_reads():
        got = [mon.score_single_token(h)["curve"].coords for h in tokens]
    assert got == want
    assert mon._curved_warm
    assert set(mon._curved_feet) == set(feet)
    for key, foot in feet.items():
        assert torch.equal(mon._curved_feet[key], foot)
