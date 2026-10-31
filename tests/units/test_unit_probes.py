"""The units read family: neuron and head probes, gates, replay, wiring.

The tap's values are checked by hand (a neuron's activation over its pole's
scale; a head's write norm, with Gemma's sandwich scale, over its scale);
the rest runs through a real tiny session: per-token loom rows, the final
aggregate, gates with and without an attached probe, the token replay, hook
lifetime and the refusals.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from drowse.generation.sampling import SamplingConfig
from drowse.model.blocks import model_spec
from drowse.model.units import UnitIndexError, head_write_matrix
from drowse.naming import HeadRef, NeuronRef
from drowse.readout.types import UNIT_ACTIVATION_OVER_SCALE
from drowse.units.capture import TapUnit, UnitTap
from drowse.units.runtime import UnitSite
from tests.support.sessions import register_vectors
from tests.support.tiny_models import tiny_model
from tests.support.units import random_unit_stats


def _site(spec: Any, ref: Any, stats: Any) -> UnitSite:
    block = spec.blocks[ref.layer]
    layer = stats.layer(ref.layer)
    if isinstance(ref, NeuronRef):
        return UnitSite(
            ref=ref, kind="neuron", layer=ref.layer, module=block.mlp.down,
            start=ref.index, stop=ref.index + 1, width=block.d_ff,
            mean=layer.neuron_mean[ref.index : ref.index + 1],
        )
    hd = block.attn.head_dim
    return UnitSite(
        ref=ref, kind="head", layer=ref.layer, module=block.attn.o,
        start=ref.index * hd, stop=(ref.index + 1) * hd,
        width=block.attn.n_heads * hd, mean=layer.head_mean[ref.index],
    )


@pytest.mark.parametrize("family", ["llama", "gemma3_text"])
def test_the_tap_reads_activation_over_scale(family: str) -> None:
    model = tiny_model(family)
    spec = model_spec(model)
    stats = random_unit_stats(spec)
    refs = [NeuronRef(1, 3), NeuronRef(1, 40), NeuronRef(2, 5), HeadRef(0, 1), HeadRef(3, 2)]
    units = []
    for ref in refs:
        layer = stats.layer(ref.layer)
        scale = (
            layer.neuron_scale[:, ref.index] if isinstance(ref, NeuronRef)
            else layer.head_scale[ref.index]
        )
        units.append(TapUnit(
            name=ref.qualified, site=_site(spec, ref, stats),
            block=spec.blocks[ref.layer], scale=scale,
        ))
    tap = UnitTap(units, device=torch.device("cpu"), dtype=torch.float32, depth=4)
    captured: dict[Any, Any] = {}

    def keep(kind: str, layer: int) -> Any:
        def hook(_m: torch.nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
            x = args[0][0, -1].clone()
            captured[(kind, layer)] = x if kind == "act" else (x, out[0, -1].clone())
        return hook

    handles = []
    for block in spec.blocks:
        handles.append(block.mlp.down.register_forward_hook(keep("act", block.layer)))
        handles.append(block.attn.o.register_forward_hook(keep("z", block.layer)))
    tap.attach()
    ids = torch.randint(3, int(model.config.vocab_size), (1, 9), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
        model(input_ids=ids[:, :5], use_cache=False)  # a second forward
    tap.detach()
    for handle in handles:
        handle.remove()
    assert tap.forwards == 2
    out = tap.values_at(1)
    assert out is not None
    values, raw = out
    for i, ref in enumerate(refs):
        block = spec.blocks[ref.layer]
        layer = stats.layer(ref.layer)
        if isinstance(ref, NeuronRef):
            act = captured[("act", ref.layer)][ref.index]
            scale = layer.neuron_scale[0 if act >= 0 else 1, ref.index]
            expected_raw = act
        else:
            z, o = captured[("z", ref.layer)]
            hd = block.attn.head_dim
            expected_raw = (
                head_write_matrix(block, ref.index) @ z[ref.index * hd : (ref.index + 1) * hd]
            ).norm()
            if block.attn_out_norm is not None:
                expected_raw = expected_raw * torch.rsqrt(o.pow(2).mean() + block.attn_out_norm.eps)
            scale = layer.head_scale[ref.index]
        torch.testing.assert_close(raw[i], expected_raw.float(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(values[i], (expected_raw / scale).float(), rtol=1e-5, atol=1e-6)
    assert tap.values_at(2) is None and tap.values_at(-1) is None


def _sampling(**kwargs: Any) -> SamplingConfig:
    return SamplingConfig(max_tokens=kwargs.pop("max_tokens", 5), seed=0, **kwargs)


def _units_channel(envelope: Any) -> Any:
    return (envelope or {}).get("instruments", {}).get("units")


def test_unit_probes_ride_the_generation(tiny_session: Any) -> None:
    session = tiny_session
    assert session.add_probe("neuron/l1N03") == "neuron/L1n3"
    assert session.add_probe("head/L2h1", as_name="h") == "h"
    assert session.units.names == ["neuron/L1n3", "h"]
    assert session.units.specs()["h"] == {
        "unit": "head/L2h1", "kind": "head", "layer": 2, "index": 1,
    }
    result = session.generate("hello", sampling=_sampling()).first
    channel = _units_channel(result.measurements)
    assert channel is not None and channel["binding"]["source"] is None
    readings = channel["readings"]
    assert set(readings) == {"neuron/L1n3", "h"}
    neuron, head = readings["neuron/L1n3"], readings["h"]
    assert neuron["unit"] == head["unit"] == UNIT_ACTIVATION_OVER_SCALE
    assert neuron["per_layer"] == {"1": neuron["value"]}
    assert neuron["depth"] == {
        "center": [round(1 / 3, 6)], "spread": [0.0], "basis": "single_layer",
    }
    assert set(neuron["meta"]) == {"unit", "activation"}
    assert set(head["meta"]) == {"unit", "write_norm"}
    assert head["value"] >= 0
    assert result.measurements["scores"]["h"] == head["value"]
    # Every loom token row carries the live readings, and the aggregate is
    # one of them (the last content token's).
    node = session.tree.nodes[session.tree.active_node_id]
    rows = [row["measurements"]["instruments"]["units"]["readings"] for row in node.tokens]
    assert len(rows) == len(node.tokens) > 0
    assert any(
        abs(row["h"]["value"] - head["value"]) < 1e-6
        and abs(row["neuron/L1n3"]["value"] - neuron["value"]) < 1e-6
        for row in rows
    )
    # The token replay rebuilds the same forward and reads the same value.
    replay = session.units.token_readout(node.id, int(node.tokens[0]["raw_index"]))
    envelope = replay["measurements"]
    assert envelope["scope"] == "replay" and envelope["provenance"] == "replayed"
    replayed = envelope["instruments"]["units"]["readings"]
    for name in ("neuron/L1n3", "h"):
        assert replayed[name]["value"] == pytest.approx(rows[0][name]["value"], abs=1e-5)
    # Hooks existed only while the generation read them.
    assert not session.units.hooks_attached
    for block in session.units.runtime.spec().blocks:
        assert not block.mlp.down._forward_hooks and not block.attn.o._forward_hooks


def test_the_tap_leaves_the_model_when_decode_ends(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hooks come off with the residual capture; the ring still serves the
    finalize aggregate."""
    plane = tiny_session._read_plane
    end = plane.end
    seen: list[tuple[bool, bool]] = []

    def spy() -> None:
        before = tiny_session.units.hooks_attached
        end()
        seen.append((before, tiny_session.units.hooks_attached))

    monkeypatch.setattr(plane, "end", spy)
    tiny_session.add_probe("neuron/L1n3")
    result = tiny_session.generate("hello", stateless=True, sampling=_sampling()).first
    assert seen[0] == (True, False)  # the decode's end; teardown may end again
    assert "neuron/L1n3" in _units_channel(result.measurements)["readings"]


def test_the_units_live_switch_drops_per_token_readings_only(tiny_session: Any) -> None:
    session = tiny_session
    session.add_probe("neuron/L1n3")
    assert session.units.set_live(False).to_dict() == {"enabled": False}
    result = session.generate("hello", sampling=_sampling()).first
    assert _units_channel(result.measurements) is not None
    node = session.tree.nodes[session.tree.active_node_id]
    assert all(
        _units_channel(row.get("measurements")) is None for row in node.tokens
    )
    with pytest.raises(TypeError):
        session.units.set_live(True, layers=[1])


def test_a_stateless_generation_without_consumers_reads_only_the_aggregate(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.units.instrument import UnitsRun

    session = tiny_session
    session.add_probe("neuron/L1n3")
    reads: list[int | None] = []
    aggregate_at = UnitsRun.aggregate_at

    def spy(self: UnitsRun, fwd: int | None) -> Any:
        reads.append(fwd)
        return aggregate_at(self, fwd)

    monkeypatch.setattr(UnitsRun, "aggregate_at", spy)
    result = session.generate("hello", stateless=True, sampling=_sampling()).first
    assert len(reads) == 1 and reads[0] is not None
    assert _units_channel(result.measurements)["readings"]["neuron/L1n3"]["unit"] == (
        UNIT_ACTIVATION_OVER_SCALE
    )


@pytest.mark.parametrize("attach", [False, True])
def test_unit_gates_fire_with_or_without_an_attached_probe(
    tiny_session: Any, attach: bool,
) -> None:
    session = tiny_session
    register_vectors(session, "v")
    if attach:
        session.add_probe("neuron/L1n3")

    def tokens(steering: str | None) -> list[int]:
        return list(session.generate(
            "hello", stateless=True, steering=steering,
            sampling=_sampling(max_tokens=8),
        ).first.tokens)

    always = tokens("2.0 v@response")
    never = tokens(None)
    assert always != never  # the push changes the sample
    assert tokens("2.0 v@when:neuron/L1n3>-1e9") == always
    assert tokens("2.0 v@when:neuron/L1n3<-1e9") == never
    assert tokens("2.0 v@when:head/L2h1>=0") == always  # a write norm is never negative
    assert not session.units.hooks_attached


def test_a_gate_on_an_unknown_unit_is_a_400_at_preflight(tiny_session: Any) -> None:
    register_vectors(tiny_session, "v")
    with pytest.raises(UnitIndexError, match=r"neuron 64 is outside block 1 \(64 neurons, 0..63\)"):
        tiny_session.generate(
            "hello", stateless=True, steering="0.5 v@when:neuron/L1n64>1",
            sampling=_sampling(),
        )
    assert not tiny_session.units.hooks_attached


@pytest.mark.parametrize(("selector", "fragment"), [
    ("neuron/L1n64", "neuron 64 is outside block 1 (64 neurons, 0..63)"),
    ("head/L9h0", "layer 9 is outside the model (4 blocks, 0..3)"),
])
def test_attaching_an_unknown_unit_is_a_400(
    tiny_session: Any, selector: str, fragment: str,
) -> None:
    with pytest.raises(UnitIndexError) as info:
        tiny_session.add_probe(selector)
    assert info.value.user_message() == (400, fragment)
    assert tiny_session.units.names == []


def test_unit_probe_hashes_track_the_scale(tiny_session: Any) -> None:
    session = tiny_session
    session.add_probe("neuron/L1n3")
    session.add_probe("neuron/L1n4")
    first = session.units.probe_hash("neuron/L1n3")
    assert first is not None and first == session.units.probe_hash("neuron/L1n3")
    assert first != session.units.probe_hash("neuron/L1n4")
    assert session.units.probe_hash("ghost") is None
    assert session.probe_hashes()["neuron/L1n3"] == first
    session.remove_probe("neuron/L1n3")
    assert session.units.names == ["neuron/L1n4"]


def test_token_readout_refuses_knobs_it_cannot_honor(tiny_session: Any) -> None:
    from drowse.errors import RequestError

    with pytest.raises(RequestError, match="top_k"):
        tiny_session.units.token_readout("n", 0, top_k=3)
    with pytest.raises(RequestError, match="layers"):
        tiny_session.units.token_readout("n", 0, layers=[1])
    assert tiny_session.units.token_readout("n", 0) == {"measurements": None}


def test_batched_generation_with_unit_probes_runs_serially(tiny_session: Any) -> None:
    tiny_session.add_probe("head/L1h0")
    runs = tiny_session.generate_batch(["one", "two"], sampling=_sampling(max_tokens=3))
    for result in runs:
        assert "head/L1h0" in _units_channel(result.measurements)["readings"]


def test_the_instruments_route_lists_the_units_family(
    tiny_session: Any, app_client: Any,
) -> None:
    client = app_client(tiny_session)
    tiny_session.add_probe("neuron/L1n3")
    info = client.get("/drowse/v1/sessions/default").json()
    families = {row["family"]: row for row in info["instruments"]}
    units = families["units"]
    assert units["probes"] == ["neuron/L1n3"]
    assert units["source"] is None
    assert units["live"] == {"enabled": True}
    assert units["capabilities"] == {
        "sources": False, "preparations": [], "token_readout": True,
        "source_switch": False,
    }
    off = client.post(
        "/drowse/v1/sessions/default/instruments/units/live", json={"enabled": False},
    )
    assert off.status_code == 200
    assert tiny_session.units.live_state.enabled is False
    assert client.get(
        "/drowse/v1/sessions/default/instruments/units/sources",
    ).status_code == 404
