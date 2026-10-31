"""The transport maps against brute-force autograd, family by family.

Every covered dense family (``tests/support/blocks.DENSE_FAMILIES``: q/k/v
biases, config and per-layer sliding windows, q/k norms, sandwich and
``1 + w`` norms, keys-as-values with a weightless value norm, per-layer
scalars, attention and final softcaps) on a tiny fp64 config with every
weight re-drawn: each kind (J, R, K, S) at both sites must equal the autograd
oracle (``tests/support/lens_oracle``) through the model's own modules.  The
floor is the models' own fp32 RMSNorm casts, about 1e-6.

Each block map's ``vjp`` is the exact transpose of its ``jvp`` (the adjoint
identity ``u·JVP(v) = VJP(u)·v``, to the block output and to ``mid``).  When
``~/Work/neuron-lens`` (or ``$NEURON_LENS_PATH``) is importable, the
transport also matches neuron-lens's hand-written Qwen3 maps (local only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.lens.key import site_layers
from drowse.lens.transport import (
    blocks_needed,
    document_estimate,
    jvp,
    linearize,
    rules_for,
    vjp,
)
from drowse.model.blocks import model_spec
from tests.support.blocks import DENSE_FAMILIES, tiny_dense_model
from tests.support.lens_oracle import oracle_estimate

_SKIP = 2
_IDS = torch.randint(3, 64, (12,), generator=torch.Generator().manual_seed(7))
#: Gemma 2's default attention cap (50) never binds on a tiny model; a cap
#: of 1 does, so its derivative is exercised.
_OVERRIDES: dict[str, dict[str, Any]] = {"gemma2": {"attn_logit_softcapping": 1.0}}


def _model(family: str) -> Any:
    return tiny_dense_model(family, **_OVERRIDES.get(family, {}))


def _all_layers(kind: str, n_layers: int) -> dict[str, list[int]]:
    return {site: site_layers(kind, site, n_layers) for site in ("out", "mid")}


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


@pytest.mark.parametrize("kind", ["J", "R", "K", "S"])
@pytest.mark.parametrize("family", sorted(DENSE_FAMILIES))
def test_document_estimate_matches_autograd(family: str, kind: str) -> None:
    model = _model(family)
    spec = model_spec(model)
    layers = _all_layers(kind, spec.n_layers)
    got = document_estimate(spec, _IDS, kind=kind, layers=layers, skip_first=_SKIP, dim_batch=7)
    want = oracle_estimate(model, _IDS, kind=kind, layers=layers, skip_first=_SKIP, batch=13)
    for site, wanted in layers.items():
        for layer in wanted:
            err = _relative(got[site][layer].cpu(), want[site][layer])
            assert err < 2e-5, (family, kind, site, layer, err)


def test_rules_differ_and_softcap_and_window_bind() -> None:
    """The fixtures exercise what they claim: RelP and standard maps differ,
    Gemma 2's softcap factor is far from 1, and a 3-wide window masks."""
    model = _model("gemma2")
    spec = model_spec(model)
    maps = linearize(spec, _IDS, rules="relp", skip_first=_SKIP)
    factor = maps[0].score_factor
    assert isinstance(factor, torch.Tensor)
    scaling = spec.blocks[0].attn.scaling
    visible = maps[0].p > 0
    assert float((factor[visible] / scaling).min()) < 0.5
    assert int(visible.sum(-1).max()) <= 3
    layers = {"out": [0], "mid": [1]}
    relp = document_estimate(spec, _IDS, kind="R", layers=layers, skip_first=_SKIP)
    standard = document_estimate(spec, _IDS, kind="J", layers=layers, skip_first=_SKIP)
    assert _relative(relp["mid"][1], standard["mid"][1]) > 1e-2


@pytest.mark.parametrize("rules", ["relp", "standard"])
@pytest.mark.parametrize("family", sorted(DENSE_FAMILIES))
def test_block_maps_satisfy_the_adjoint_identity(family: str, rules: str) -> None:
    spec = model_spec(_model(family))
    maps = linearize(spec, _IDS, rules=rules, skip_first=_SKIP)  # type: ignore[arg-type]
    gen = torch.Generator().manual_seed(3)
    for layer_map in maps.values():
        shape = (3, layer_map.n_rows, spec.d_model)
        u = torch.randn(shape, generator=gen, dtype=torch.float64)
        v = torch.randn(shape, generator=gen, dtype=torch.float64)
        out_v, mid_v = jvp(layer_map, v)
        inp_u, _mid_u = vjp(layer_map, u)
        lhs, rhs = float((u * out_v).sum()), float((inp_u * v).sum())
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0), (family, layer_map.layer)
        # The map to ``mid`` and its transpose (a cotangent seeded there).
        inp_mid, _ = vjp(layer_map, torch.zeros_like(u), mid_seed=u)
        lhs, rhs = float((u * mid_v).sum()), float((inp_mid * v).sum())
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0), (family, layer_map.layer)


def test_restricted_layers_match_the_full_sweep() -> None:
    spec = model_spec(_model("qwen3"))
    full_out = document_estimate(spec, _IDS, kind="R", layers=_all_layers("R", 3), skip_first=_SKIP)
    part = document_estimate(spec, _IDS, kind="R", layers={"out": [1]}, skip_first=_SKIP)
    assert set(part) == {"out"} and set(part["out"]) == {1}
    assert torch.allclose(part["out"][1], full_out["out"][1], atol=1e-12)
    full_in = document_estimate(spec, _IDS, kind="S", layers=_all_layers("S", 3), skip_first=_SKIP)
    part = document_estimate(spec, _IDS, kind="S", layers={"mid": [0]}, skip_first=_SKIP)
    assert torch.allclose(part["mid"][0], full_in["mid"][0], atol=1e-12)


def test_sweeps_touch_only_the_blocks_they_need() -> None:
    assert blocks_needed("output", 6, {"out": [3], "mid": [4]}) == [4, 5]
    assert blocks_needed("output", 6, {"mid": [2]}) == [2, 3, 4, 5]
    assert blocks_needed("input", 6, {"out": [1], "mid": [3]}) == [0, 1, 2, 3]
    assert blocks_needed("input", 6, {}) == []
    assert rules_for("R") == rules_for("S") == "relp"
    assert rules_for("J") == rules_for("K") == "standard"
    with pytest.raises(ValueError):
        rules_for("Q")


def test_linearize_refuses_a_document_without_valid_positions() -> None:
    spec = model_spec(_model("llama"))
    with pytest.raises(ValueError, match="no valid position"):
        linearize(spec, _IDS[: _SKIP + 1], rules="relp", skip_first=_SKIP)


# -- agreement with neuron-lens (local only) -------------------------------------------

def _neuron_lens() -> Any:
    root = Path(os.environ.get("NEURON_LENS_PATH", Path.home() / "Work" / "neuron-lens"))
    if not (root / "neuron_lens" / "transport.py").exists():
        pytest.skip("neuron-lens is not checked out on this machine")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import neuron_lens.transport as transport
    except ImportError as exc:  # pragma: no cover - a broken checkout
        pytest.skip(f"neuron-lens does not import: {exc}")
    return transport


@pytest.mark.parametrize("kind", ["R", "S"])
def test_transport_matches_neuron_lens_on_tiny_qwen3(kind: str) -> None:
    nl = _neuron_lens()
    model = tiny_dense_model("qwen3").float()
    spec = model_spec(model)
    ids = torch.randint(3, 64, (nl.SKIP_FIRST + 9,), generator=torch.Generator().manual_seed(11))
    n = spec.n_layers
    sites = {"out": n - 1, "mid": n} if kind == "R" else {"out": n, "mid": n}
    acc = nl.Accumulator(sites, spec.d_model)
    nl.lens_doc(kind, model.model, ids, acc, 0, dim_batch=5)
    ours = document_estimate(
        spec, ids, kind=kind, layers=_all_layers(kind, n), skip_first=nl.SKIP_FIRST,
    )
    for site, count in sites.items():
        theirs = acc.mean(site)
        for layer in range(count):
            err = _relative(ours[site][layer], theirs[layer])
            assert err < 1e-5, (kind, site, layer, err)
