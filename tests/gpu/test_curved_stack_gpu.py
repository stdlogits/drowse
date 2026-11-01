"""On the accelerator, a curved probe's stacked read is the per-layer read, bit for bit.

The stacked read batches only work whose batched kernel computes each layer as
the one-layer kernel does (elementwise ops, the foot solve's batched products)
and keeps every reduction on its own layer's rows; the cross-layer means fold
in layer order.  This pins that on the device, against the per-layer read
written out (each layer's projection, ``invert_parameterization`` and σ, then
sequential weighted sums), so batching a reduction across layers (which moves
values by an ulp, and the foot of an off-surface state to another basin) fails
here.  Exact on MPS; CUDA's batched and single kernels are not checked for
per-slice identity, so there the comparison takes a tolerance.
"""
from __future__ import annotations

import pytest
import torch

from drowse.geometry.invert import invert_parameterization
from drowse.readout.monitor import Monitor, _mean_coordinate_space, _mean_domain_coordinates
from tests.support.curved import DIM, curved_stack_manifold as _curved
from tests.support.whitener import isotropic_whitener

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
EXACT = DEVICE.type == "mps"


def _per_layer(mon: Monitor, hidden: dict[int, torch.Tensor], feet: dict[int, torch.Tensor] | None):
    """The per-layer read, as the Monitor ran it before the stacked pass."""
    probe = mon.attached_probes()["curve"]
    m = probe.manifold
    stack = mon._read_stack(DEVICE)
    H = torch.stack([hidden[L].float() for L in stack.layers]).unsqueeze(1)
    sih = stack.apply_stack(H)
    shared = list(m.layers)
    total = sum(probe.share_weights[L] for L in shared)
    w = {L: probe.share_weights[L] / total for L in shared}
    ref_layer = max(shared, key=lambda L: w[L])
    frac_mean = coords_mean = mem_mean = None
    ref_coord = None
    new_feet = {}
    for L in shared:
        wh, sub, i = probe.whitened[L], m.layers[L], stack.index[L]
        x = H[i, 0] - wh.mean
        sx = sih[i, 0] - wh.s_mean
        x_m = torch.sqrt((x * sx).sum().clamp(min=0.0)).clamp(min=1e-8)
        g = wh.basis @ sx
        c = wh.m_r_inv @ g
        frac = (torch.sqrt((g * c).sum().clamp(min=0.0)) / x_m).clamp(min=0.0, max=1.0).reshape(1)
        pos, res = invert_parameterization(
            sub, m.domain, c.reshape(1, -1), m.node_coords,
            warm_start=None if feet is None else feet[L],
        )
        new_feet[L] = pos
        coord = pos.reshape(-1)
        if L == ref_layer:
            ref_coord = coord
        space = w[L] * _mean_coordinate_space(m.domain, coord)
        sig = sub.sigma_at(m.domain.embed(pos.reshape(1, -1))).reshape(1)
        mem = torch.exp(-(res.reshape(-1)[:1] ** 2) / (2.0 * (sig ** 2).clamp(min=1e-8)))
        frac_mean = w[L] * frac if frac_mean is None else frac_mean + w[L] * frac
        coords_mean = space if coords_mean is None else coords_mean + space
        mem_mean = w[L] * mem if mem_mean is None else mem_mean + w[L] * mem
    assert ref_coord is not None and coords_mean is not None
    assert frac_mean is not None and mem_mean is not None
    coords = _mean_domain_coordinates(m.domain, coords_mean, ref_coord)
    return (float(frac_mean), coords.tolist(), float(mem_mean)), new_feet


@pytest.mark.parametrize("periodic", [False, True])
def test_stacked_curved_read_is_the_per_layer_read(periodic: bool) -> None:
    m = _curved(n_layers=6, periodic=periodic)
    mon = Monitor(whitener=isotropic_whitener(list(m.layers), DIM), n_layers=8)
    mon.add_probe("curve", m.to(device=DEVICE, dtype=torch.float32), top_n=3)
    gen = torch.Generator().manual_seed(11)
    mon.reset_curved_feet()
    mon.enable_curved_warm(True)
    feet = None
    with torch.inference_mode():
        for _ in range(8):
            hidden = {}
            for L, sub in m.layers.items():
                t = torch.rand((1, 1), generator=gen) * (6.0 if periodic else 1.6) - (0.0 if periodic else 0.8)
                on = sub.eval_at(m.domain.embed(t)).reshape(-1)
                hidden[L] = (on + 0.3 * torch.randn(DIM, generator=gen)).to(DEVICE)
            got = mon.score_single_token(hidden)["curve"]
            (frac, coords, mem), feet = _per_layer(mon, hidden, feet)
            if EXACT:
                assert (got.fraction, list(got.coords), got.membership) == (frac, coords, mem)
            else:
                assert got.fraction == pytest.approx(frac, abs=1e-5)
                assert list(got.coords) == pytest.approx(coords, abs=1e-4)
                assert got.membership == pytest.approx(mem, abs=1e-5)
