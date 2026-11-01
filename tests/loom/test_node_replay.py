"""The node replay: token anatomy and unit activity over one tapped forward.

Pinned against hand computations on a tiny session: the taps decompose the
block output exactly (block input + attention write + MLP write + steering
injection), every lens read is the one ``plan_reads`` gives for the same
vector, the unit contributions equal a brute-force autograd Jacobian of the
lens-linearized logit, and unit activity equals the units family's own probe
readings at the same tokens.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from drowse.generation.sampling import SamplingConfig
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.loom.node_replay import (
    POPULATION_ENCODING,
    decode_population,
    eager_attention,
    tap_token,
    unembed_cotangent,
)
from drowse.model.blocks import model_spec

pytestmark = pytest.mark.usefixtures("isolated_home")


def _install_lens(session: Any, kind: str, *, seed: int) -> JacobianLens:
    """Publish (and select) a local lens of ``kind`` at both sites for the
    session's exact weights: every layer, except that an output kind's ``out``
    stops below the last block, as a fit's does (its transport there is the
    identity)."""
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.model.fingerprint import loaded_model_fingerprint

    n = len(session.layers)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(seed)
    mats = {L: torch.eye(d) + 0.2 * torch.randn(d, d, generator=gen) for L in range(n)}
    out = {L: m for L, m in mats.items() if not (LensKey(kind).side == "output" and L == n - 1)}
    lens = JacobianLens(out, mid=dict(mats), n_prompts=3, d_model=d, key=LensKey(kind))
    save_lens(lens, session.model_id, LensProvenance(
        corpus_spec="test", corpus_sha256="a" * 64, seq_len=8, dim_batch=1,
        model_fingerprint=loaded_model_fingerprint(session.model, session.model_id),
    ))
    return lens


def _node(session: Any, *, steering: str | None = None, max_tokens: int = 6) -> Any:
    session.generate(
        "hello", steering=steering, sampling=SamplingConfig(max_tokens=max_tokens, seed=0),
    )
    return session.tree.nodes[session.tree.active_node_id]


# -- the taps ---------------------------------------------------------------------


def test_the_taps_decompose_each_block_output(tiny_session: Any) -> None:
    from drowse.fit.capture import _capture_all_hidden_states
    from tests.support.sessions import register_vectors

    session = tiny_session
    spec = model_spec(session.model)
    ids = session.tokenizer("the cat sat", return_tensors="pt")["input_ids"]
    p = int(ids.shape[1]) - 1
    taps = tap_token(session.model, spec, ids, p, patterns=False)
    hidden = _capture_all_hidden_states(session.model, session.layers, ids)
    for L in range(len(spec.blocks)):
        torch.testing.assert_close(taps.out[L], hidden[L][0, p].float(), rtol=1e-5, atol=1e-6)
    whole = taps.block_in + taps.attention_write + taps.mlp_write + taps.steering_write
    torch.testing.assert_close(whole, taps.out, rtol=0, atol=1e-6)
    assert float(taps.steering_write.abs().max()) == 0.0
    # The block input is the previous block's output (and the scaled
    # embedding at block 0).
    torch.testing.assert_close(taps.block_in[1:], taps.out[:-1])
    embed = spec.embed.weight[ids[0, p]].float() * spec.embed_scale
    torch.testing.assert_close(taps.block_in[0], embed)
    # Steering lands between the two block-output taps, nowhere else.
    register_vectors(session, "probe.axis")
    with session.steering("0.5 probe.axis"):
        steered = tap_token(session.model, spec, ids, p, patterns=False)
    injected = steered.steering_write.abs().amax(dim=1)
    assert bool((injected > 0).any())
    whole = steered.block_in + steered.attention_write + steered.mlp_write + steered.steering_write
    torch.testing.assert_close(whole, steered.out, rtol=0, atol=1e-6)


def test_patterns_are_the_attention_probabilities_under_eager(tiny_session: Any) -> None:
    session = tiny_session
    spec = model_spec(session.model)
    config: Any = cast(Any, spec.blocks[0].attn.module).config
    ids = session.tokenizer("abcdef", return_tensors="pt")["input_ids"]
    p = int(ids.shape[1]) - 1
    before = config._attn_implementation
    config._attn_implementation = "sdpa"
    try:
        with eager_attention(spec, True):
            assert config._attn_implementation == "eager"
            taps = tap_token(session.model, spec, ids, p, patterns=True)
        assert config._attn_implementation == "sdpa"
    finally:
        config._attn_implementation = before
    assert taps.patterns is not None
    for rows in taps.patterns:
        assert rows.shape == (spec.blocks[0].attn.n_heads, p + 1)
        torch.testing.assert_close(rows.sum(dim=1), torch.ones(rows.shape[0]))


# -- token anatomy ------------------------------------------------------------------


def _brute_neurons(
    session: Any, L: int, taps: Any, matrix: torch.Tensor, u: torch.Tensor,
) -> torch.Tensor:
    """``act ⊙ ∂(u · J (mid + down(act)))/∂act`` by autograd through the
    model's own down projection."""
    block = model_spec(session.model).blocks[L]
    act = taps.acts[L].clone().requires_grad_(True)
    with torch.enable_grad():
        out = taps.mid[L] + block.mlp.down(act)
        f = u @ (matrix @ out)
        (grad,) = torch.autograd.grad(f, act)
    return (grad * taps.acts[L]).detach()


def _brute_heads(
    session: Any, L: int, taps: Any, matrix: torch.Tensor, u: torch.Tensor,
) -> torch.Tensor:
    block = model_spec(session.model).blocks[L]
    z = taps.z[L].clone().requires_grad_(True)
    with torch.enable_grad():
        mid = taps.block_in[L] + block.attn.o(z)
        f = u @ (matrix @ mid)
        (grad,) = torch.autograd.grad(f, z)
    per = (grad * taps.z[L]).detach()
    return per.reshape(block.attn.n_heads, block.attn.head_dim).sum(dim=1)


def _spy_taps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, Any]]:
    """Record the ids and taps every anatomy forward used."""
    import drowse.loom.node_replay as node_replay

    seen: list[tuple[torch.Tensor, Any]] = []
    original = node_replay.tap_token

    def spy(model: Any, spec: Any, ids: torch.Tensor, position: int, **kwargs: Any) -> Any:
        taps = original(model, spec, ids, position, **kwargs)
        seen.append((ids.clone(), taps))
        return taps

    monkeypatch.setattr(node_replay, "tap_token", spy)
    return seen


def test_token_anatomy_reads_every_layer_and_its_units(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tiny_session
    r_lens = _install_lens(session, "R", seed=1)
    _install_lens(session, "S", seed=2)
    node = _node(session)
    raw_index = 2
    seen = _spy_taps(monkeypatch)
    envelope = session.token_anatomy(node.id, raw_index, k=3, units=4, patterns=2)
    m = envelope["measurements"]
    assert m["scope"] == "replay" and m["provenance"] == "replayed"
    assert m["instruments"] == {} and set(m) == {
        "version", "scope", "provenance", "instruments", "anatomy",
    }
    anatomy = m["anatomy"]
    n = len(session.layers)
    token = anatomy["token"]
    assert token["raw_index"] == raw_index and token["token_id"] == node.raw_token_ids[raw_index]
    assert token["position"] == token["prompt_length"] + raw_index - 1
    assert anatomy["binding"] == {"sources": {"R": "local:r", "S": "local:s"}, "steering": None}
    assert [layer["layer"] for layer in anatomy["layers"]] == list(range(n))

    # The producing forward: the prompt render, then the decode record up to
    # the token; every row is held to a plain read of the tapped vectors.
    from drowse.lens.read import VectorRead

    spec = model_spec(session.model)
    (ids, taps), = seen
    assert int(ids.shape[1]) == token["prompt_length"] + raw_index == token["position"] + 1
    assert ids[0, token["prompt_length"]:].tolist() == list(node.raw_token_ids[:raw_index])
    with torch.inference_mode():
        pending = session.lens.plan_reads([
            VectorRead(taps.out[1], 1, "out", "residual"),
            VectorRead(taps.attention_write[1], 1, "mid", "direction"),
            VectorRead(taps.mlp_write[1], 1, "out", "direction"),
        ], k=3)
        expected = pending.finish(pending.tensor.cpu())
    layer1 = anatomy["layers"][1]
    for key, (read,) in zip(("reads_as", "attention", "mlp"), expected, strict=True):
        got = layer1[key]
        assert got["site"] == read.site and got["of"] == read.of
        assert [t["token_id"] for t in got["s"]["tokens"]] == [t.token_id for t in read.s.tokens]
        assert [t["token_id"] for t in got["r"]["tokens"]] == [t.token_id for t in read.r.tokens]
    assert layer1["reads_as"]["of"] == "residual" and layer1["reads_as"]["s"]["centred"]
    assert layer1["reads_as"]["s"]["lens"] == "S:out" and layer1["reads_as"]["r"]["lens"] == "R:out"
    assert layer1["attention"]["site"] == "mid" and not layer1["attention"]["s"]["centred"]
    assert layer1["attention"]["s"]["lens"] == "S:mid"
    assert all(layer["steering"] is None for layer in anatomy["layers"])

    # The last block's out: the R lens stops a layer short there, and its
    # transport is the identity, so the residual and the MLP write read through
    # the logit lens (labelled ``logit``); the attention write keeps R:mid.
    last = anatomy["layers"][n - 1]
    for key in ("reads_as", "mlp"):
        assert last[key]["r"]["absent"] is None and last[key]["r"]["lens"] == "logit"
    assert last["attention"]["r"]["lens"] == "R:mid"
    with torch.inference_mode():
        logits = session.lens.runtime.readout_modules()[1](taps.out[-1][None].float())[0].float()
    assert [t["token_id"] for t in last["reads_as"]["r"]["tokens"]] == logits.softmax(-1).topk(3).indices.tolist()

    # Contributions against a brute-force Jacobian of the linearized logit.
    v = int(token["token_id"])
    h = taps.out[-1].clone().requires_grad_(True)
    with torch.enable_grad():
        norm = spec.final_norm
        r = torch.rsqrt(h.pow(2).mean() + norm.eps).detach()
        logit = (h * r * norm.gain()) @ spec.unembed[v].float()
        (u_brute,) = torch.autograd.grad(logit, h)
    u = unembed_cotangent(spec, taps.out[-1], v)
    torch.testing.assert_close(u, u_brute)
    unit_stats = session.units.runtime.stats()
    for L, layer in enumerate(anatomy["layers"]):
        assert 0 < len(layer["neurons"]) <= 4 and 0 < len(layer["heads"]) <= 4
        J_out = torch.eye(u.shape[0]) if L == n - 1 else r_lens.matrix(L, "out")
        neurons = _brute_neurons(session, L, taps, J_out, u)
        heads = _brute_heads(session, L, taps, r_lens.matrix(L, "mid"), u)
        top_n = neurons.abs().topk(len(layer["neurons"])).indices.tolist()
        assert [unit["index"] for unit in layer["neurons"]] == top_n
        for unit in layer["neurons"]:
            assert unit["unit"] == f"neuron/L{L}n{unit['index']}" and unit["kind"] == "neuron"
            assert unit["contribution"] == pytest.approx(float(neurons[unit["index"]]), abs=1e-5)
            assert unit["activation"] == pytest.approx(float(taps.acts[L][unit["index"]]), abs=1e-5)
            # The units family's number: the activation over the pole's soft scale.
            pole = 0 if unit["activation"] >= 0 else 1
            scale = float(unit_stats.layer(L).neuron_reading_scale[pole, unit["index"]])
            expected = float(taps.acts[L][unit["index"]]) / scale
            assert unit["activation_over_scale"] == pytest.approx(expected, abs=1e-5)
        for unit in layer["heads"]:
            assert unit["unit"] == f"head/L{L}h{unit['index']}"
            assert unit["contribution"] == pytest.approx(float(heads[unit["index"]]), abs=1e-5)
            assert unit["activation"] >= 0
            scale = float(unit_stats.layer(L).head_reading_scale[unit["index"]])
            assert unit["activation_over_scale"] == pytest.approx(unit["activation"] / scale, rel=1e-4, abs=1e-5)
    # The cross-layer list is the largest |contribution| of every unit.
    every = sorted(
        (abs(u["contribution"]) for layer in anatomy["layers"] for u in layer["neurons"] + layer["heads"]),
        reverse=True,
    )[:4]
    assert [abs(u["contribution"]) for u in anatomy["units"]] == pytest.approx(every, abs=1e-6)
    assert anatomy["units_absent"] is None
    # Patterns: the top heads by |contribution|, over the whole context.
    patterns = anatomy["patterns"]
    assert patterns is not None and len(patterns) == 2
    for pattern in patterns:
        assert len(pattern["weights"]) == token["position"] + 1
        assert sum(pattern["weights"]) == pytest.approx(1.0, abs=1e-4)
    assert anatomy["context"] is not None and len(anatomy["context"]) == token["position"] + 1


def test_sandwich_blocks_hold_their_output_norms_scale(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma's post-attention and post-feedforward norms: a unit's write
    passes the norm's gain and its per-position scale, held constant (the LN
    rule); the contributions match autograd with that scale detached, and
    activity matches the units family's probes."""
    from tests.support.sessions import TinyCheckpoint, TinySessionFactory

    factory = TinySessionFactory(TinyCheckpoint.write(tmp_path, "gemma3_text"), tmp_path / "seed")
    try:
        session = factory()
        spec = model_spec(session.model)
        assert spec.blocks[0].mlp_out_norm is not None and spec.blocks[0].attn_out_norm is not None
        r_lens = _install_lens(session, "R", seed=4)
        session.add_probe("head/L1h0")
        node = _node(session, max_tokens=4)
        seen = _spy_taps(monkeypatch)
        anatomy = session.token_anatomy(node.id, 1, units=3, patterns=0)["measurements"]["anatomy"]
        (_ids, taps), = seen
        u = unembed_cotangent(spec, taps.out[-1], int(anatomy["token"]["token_id"]))
        for L, layer in enumerate(anatomy["layers"]):
            block = spec.blocks[L]
            act = taps.acts[L].clone().requires_grad_(True)
            z = taps.z[L].clone().requires_grad_(True)
            with torch.enable_grad():
                x = block.mlp.down(act)
                r = torch.rsqrt(x.pow(2).mean() + block.mlp_out_norm.eps).detach()  # type: ignore[union-attr]
                out = (taps.mid[L] + x * r * block.mlp_out_norm.gain()) * block.scale()  # type: ignore[union-attr]
                J_out = torch.eye(u.shape[0]) if L == len(spec.blocks) - 1 else r_lens.matrix(L, "out")
                (g_act,) = torch.autograd.grad(u @ (J_out @ out), act)
                o = block.attn.o(z)
                r = torch.rsqrt(o.pow(2).mean() + block.attn_out_norm.eps).detach()  # type: ignore[union-attr]
                mid = taps.block_in[L] + o * r * block.attn_out_norm.gain()  # type: ignore[union-attr]
                (g_z,) = torch.autograd.grad(u @ (r_lens.matrix(L, "mid") @ mid), z)
            neurons = (g_act * taps.acts[L]).detach()
            heads = (g_z * taps.z[L]).detach().reshape(block.attn.n_heads, -1).sum(dim=1)
            for unit in layer["neurons"]:
                assert unit["contribution"] == pytest.approx(float(neurons[unit["index"]]), abs=1e-5)
            for unit in layer["heads"]:
                assert unit["contribution"] == pytest.approx(float(heads[unit["index"]]), abs=1e-5)
        activity = session.unit_activity(node.id, units=["head/L1h0"])
        for raw_index in range(len(node.raw_token_ids)):
            replay = session.units.token_readout(node.id, raw_index)["measurements"]
            value = replay["instruments"]["units"]["readings"]["head/L1h0"]["value"]
            assert activity["traces"]["head/L1h0"][raw_index] == pytest.approx(value, abs=1e-5)
    finally:
        factory.close()


def test_anatomy_without_an_r_lens_reads_but_has_no_units(tiny_session: Any) -> None:
    session = tiny_session
    node = _node(session)
    anatomy = session.token_anatomy(node.id, 0, patterns=3)["measurements"]["anatomy"]
    assert anatomy["units"] == [] and anatomy["patterns"] is None
    assert "R-family lens" in str(anatomy["units_absent"])
    layer = anatomy["layers"][0]
    assert layer["neurons"] == [] and layer["heads"] == []
    assert layer["reads_as"]["r"]["absent"] is not None
    assert anatomy["context"] is None


def test_anatomy_under_steering_reads_the_injection(tiny_session: Any) -> None:
    from tests.support.sessions import register_vectors

    session = tiny_session
    _install_lens(session, "R", seed=3)
    register_vectors(session, "probe.axis")
    node = _node(session, steering="0.5 probe.axis")
    steered = session.token_anatomy(node.id, 1, patterns=0)["measurements"]
    anatomy = steered["anatomy"]
    assert anatomy["binding"]["steering"] == "0.5 probe.axis"
    rows = [layer["steering"] for layer in anatomy["layers"]]
    assert any(row is not None for row in rows)
    for row in rows:
        if row is not None:
            assert row["of"] == "direction" and row["site"] == "out"
    unsteered = session.token_anatomy(node.id, 1, patterns=0, apply_steering=False)
    assert all(layer["steering"] is None for layer in unsteered["measurements"]["anatomy"]["layers"])
    assert unsteered["measurements"]["anatomy"]["binding"]["steering"] is None


def test_anatomy_refuses_bad_knobs_and_anchors(tiny_session: Any) -> None:
    from drowse.errors import RequestError
    from drowse.loom.errors import InvalidNodeOperationError

    session = tiny_session
    node = _node(session)
    for kwargs in ({"k": 0}, {"units": 65}, {"patterns": 17}):
        with pytest.raises(RequestError):
            session.token_anatomy(node.id, 0, **kwargs)
    with pytest.raises(InvalidNodeOperationError):
        session.token_anatomy(node.id, 999)


def test_anatomy_without_an_r_mid_layer_leaves_attention_alone(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: eager attention and the pattern hooks ran whenever
    ``patterns > 0``, even with no R lens to rank heads by (no pattern can
    be returned)."""
    import drowse.loom.node_replay as node_replay

    session = tiny_session
    node = _node(session)
    enabled: list[bool] = []
    original = node_replay.eager_attention

    def spy(spec: Any, on: bool) -> Any:
        enabled.append(on)
        return original(spec, on)

    monkeypatch.setattr(node_replay, "eager_attention", spy)
    seen = _spy_taps(monkeypatch)
    session.token_anatomy(node.id, 1, patterns=3)
    assert enabled == [False] and seen[0][1].patterns is None
    _install_lens(session, "R", seed=5)
    anatomy = session.token_anatomy(node.id, 1, patterns=3)["measurements"]["anatomy"]
    assert enabled == [False, True] and anatomy["patterns"] is not None


def _host_reads(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[int]]:
    """Record the storage of every pack the replay read back and of every
    tensor read into a Python list while it ran."""
    import drowse.loom.node_replay as node_replay

    packs: list[int] = []
    lists: list[int] = []
    host = node_replay._Pack.host
    tolist = torch.Tensor.tolist

    def counting_host(self: Any) -> torch.Tensor:
        out = host(self)
        packs.append(out.untyped_storage().data_ptr())
        return out

    def counting_tolist(self: torch.Tensor) -> Any:
        lists.append(self.untyped_storage().data_ptr())
        return tolist(self)

    monkeypatch.setattr(node_replay._Pack, "host", counting_host)
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    return packs, lists


def test_anatomy_and_unit_activity_read_back_one_pack(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: anatomy with patterns read the context ids back apart from
    its pack, and unit activity read its sink mask back.  Every host read
    is now a view of the one packed transfer."""
    session = tiny_session
    _install_lens(session, "R", seed=1)
    node = _node(session)
    expected_tail = [session.tokenizer.decode([int(t)]) for t in node.raw_token_ids[:2]]
    with monkeypatch.context() as patch:
        packs, lists = _host_reads(patch)
        anatomy = session.token_anatomy(node.id, 2, patterns=2)["measurements"]["anatomy"]
    assert len(packs) == 1 and lists and set(lists) == set(packs)
    context = anatomy["context"]
    assert context is not None and context[-2:] == expected_tail
    with monkeypatch.context() as patch:
        packs, lists = _host_reads(patch)
        activity = session.unit_activity(node.id, top=4, units=["neuron/L1n3"], raw_index=1)
    assert len(packs) == 1 and set(lists) <= set(packs)
    assert activity["sink_raw_index"] is None


# -- triggers --------------------------------------------------------------------------


def test_replay_segments_follow_each_trigger_window() -> None:
    from drowse.loom.replay import ReplaySegment, replay_segments
    from drowse.steering.triggers import Trigger

    def spans(triggers: Any) -> list[tuple[int, int, bool, int]]:
        return [
            (seg.start, seg.stop, seg.prefill, seg.gen_step)
            for seg in replay_segments(5, 9, triggers, can_think=False)
        ]

    # Every term fires everywhere: one forward, whatever the trigger state.
    assert replay_segments(5, 9, [], can_think=True) == (ReplaySegment(0, 9, False, 0),)
    assert spans([Trigger.BOTH, Trigger()]) == [(0, 9, False, 0)]
    # The prompt is the prefill; decode step k sits at position 5 + k - 1.
    assert spans([Trigger.PROMPT_ONLY]) == [(0, 5, True, 0), (5, 9, False, 1)]
    assert spans([Trigger.GENERATED_ONLY]) == [(0, 5, True, 0), (5, 9, False, 1)]
    assert spans([Trigger.first(2)]) == [(0, 5, True, 0), (5, 6, False, 1), (6, 9, False, 2)]
    assert spans([Trigger.after(3), Trigger.first(2)]) == [
        (0, 5, True, 0), (5, 6, False, 1), (6, 7, False, 2), (7, 9, False, 3),
    ]
    # A family that cannot think is never in a thinking section.
    assert spans([Trigger.AFTER_THINKING]) == spans([Trigger.GENERATED_ONLY])
    # Raw token 0's anatomy is the prefill alone.
    assert replay_segments(5, 5, [Trigger.GENERATED_ONLY], can_think=False) == (
        ReplaySegment(0, 5, True, 0),
    )


def test_replay_refuses_triggers_the_node_does_not_record() -> None:
    from dataclasses import replace

    from drowse.errors import SteeringCompositionError
    from drowse.loom.replay import replay_segments
    from drowse.steering.triggers import Trigger

    with pytest.raises(SteeringCompositionError, match="probe-gated") as info:
        replay_segments(5, 9, [Trigger.when("x", ">", 0.1)], can_think=False)
    assert info.value.user_message()[0] == 422
    for phase in (Trigger.AFTER_THINKING, Trigger.THINKING_ONLY):
        with pytest.raises(SteeringCompositionError, match="thinking-phase"):
            replay_segments(5, 9, [phase], can_think=True)
    # A gate on a prompt-only term never fires: nothing to reproduce.
    never = replace(Trigger.PROMPT_ONLY, gate=Trigger.when("x", ">", 0.1).gate)
    assert len(replay_segments(5, 9, [never], can_think=True)) == 2


@pytest.mark.parametrize("window", ["@prompt", "@response", "@first:2", "@after:2"])
def test_a_triggered_node_replays_as_generation_steered_it(
    tiny_session: Any, window: str,
) -> None:
    """Regression: the replay ran one forward at the trigger context's rest
    state, so a prompt-only term never fired, a response term fired on the
    prompt too and token windows were wrong.  Each token's unit activity now
    equals the live reading generation recorded, and the anatomy's steering
    row is present exactly where the term fired."""
    from tests.support.sessions import register_vectors

    session = tiny_session
    register_vectors(session, "probe.axis")
    session.add_probe("neuron/L1n3")
    session.add_probe("head/L2h1")
    node = _node(session, steering=f"2 probe.axis{window}", max_tokens=5)
    count = len(node.raw_token_ids)
    names = ["neuron/L1n3", "head/L2h1"]
    activity = session.unit_activity(node.id, units=names)
    assert node.recipe is not None and activity["steering"] == node.recipe.steering
    rows = {
        int(row["raw_index"]): row["measurements"]["instruments"]["units"]["readings"]
        for row in node.tokens
    }
    assert rows
    for raw_index, readings in rows.items():
        for name, trace in activity["traces"].items():
            assert trace[raw_index] == pytest.approx(readings[name]["value"], abs=1e-4)
    unsteered = session.unit_activity(node.id, units=names, apply_steering=False)
    assert unsteered["traces"] != activity["traces"]
    fired = {
        "@prompt": lambda k: k == 0,
        "@response": lambda k: k >= 1,
        "@first:2": lambda k: 1 <= k < 2,
        "@after:2": lambda k: k >= 2,
    }[window]
    for raw_index in range(count):
        anatomy = session.token_anatomy(node.id, raw_index, patterns=0)
        layers = anatomy["measurements"]["anatomy"]["layers"]
        injected = any(layer["steering"] is not None for layer in layers)
        assert injected == fired(raw_index), raw_index


def test_a_gated_open_steering_refuses_the_replay(tiny_session: Any) -> None:
    from drowse.errors import SteeringCompositionError
    from tests.support.sessions import register_vectors

    session = tiny_session
    register_vectors(session, "probe.axis")
    node = _node(session, max_tokens=3)
    with session.steering("0.5 probe.axis@when:neuron/L1n3 > 0"):
        with pytest.raises(SteeringCompositionError, match="probe-gated"):
            session.unit_activity(node.id)
        with pytest.raises(SteeringCompositionError, match="probe-gated"):
            session.token_anatomy(node.id, 1, patterns=0)
    # The session replays again once the scope closes.
    session.unit_activity(node.id)


# -- unit activity -------------------------------------------------------------------


def test_unit_activity_equals_the_unit_probes_at_the_same_tokens(tiny_session: Any) -> None:
    session = tiny_session
    session.add_probe("neuron/L1n3")
    session.add_probe("head/L2h1")
    node = _node(session, max_tokens=5)
    count = len(node.raw_token_ids)
    activity = session.unit_activity(
        node.id, top=6, units=["neuron/L1n3", "head/L2h1"], raw_index=2,
    )
    assert activity["node_id"] == node.id and activity["unit"] == "activation_over_scale"
    assert [t["raw_index"] for t in activity["tokens"]] == list(range(count))
    assert activity["sink_raw_index"] is None
    traces = activity["traces"]
    assert set(traces) == {"neuron/L1n3", "head/L2h1"}
    for raw_index in range(count):
        replay = session.units.token_readout(node.id, raw_index)["measurements"]
        readings = replay["instruments"]["units"]["readings"]
        for name in traces:
            assert traces[name][raw_index] == pytest.approx(readings[name]["value"], abs=1e-5)
    # The live loom rows read the same values.
    rows = {
        int(row["raw_index"]): row["measurements"]["instruments"]["units"]["readings"]
        for row in node.tokens
    }
    for raw_index, readings in rows.items():
        for name in traces:
            assert traces[name][raw_index] == pytest.approx(readings[name]["value"], abs=1e-4)
    # Peaks: each is its unit's largest |value| over the node, at its token.
    top = activity["top"]
    assert len(top) == 6
    assert [abs(p["peak"]) for p in top] == sorted((abs(p["peak"]) for p in top), reverse=True)
    population = activity["population"]
    assert population is not None and population["encoding"] == POPULATION_ENCODING
    spec = model_spec(session.model)
    assert [layer["layer"] for layer in population["layers"]] == list(range(len(spec.blocks)))
    full = {
        layer["layer"]: (decode_population(layer["neurons"]), decode_population(layer["heads"]))
        for layer in population["layers"]
    }
    assert full[1][0].shape == (spec.blocks[1].d_ff,) and full[2][1].shape == (spec.blocks[2].n_heads,)
    assert float(full[1][0][3]) == pytest.approx(traces["neuron/L1n3"][2], rel=2e-3, abs=2e-3)
    assert float(full[2][1][1]) == pytest.approx(traces["head/L2h1"][2], rel=2e-3, abs=2e-3)
    listed = {(p["kind"], p["layer"], p["index"]) for p in top}
    for peak in top:
        if peak["raw_index"] == 2:
            values = full[peak["layer"]][0 if peak["kind"] == "neuron" else 1]
            assert float(values[peak["index"]]) == pytest.approx(peak["peak"], rel=2e-3, abs=2e-3)
    # A unit off the list never fires harder at token 2 than the list's last
    # peak (its own peak is no larger).
    smallest = abs(top[-1]["peak"])
    for layer, (neurons, heads) in full.items():
        for kind, values in (("neuron", neurons), ("head", heads)):
            for index, value in enumerate(values.tolist()):
                if (kind, layer, index) not in listed:
                    assert abs(value) <= smallest * (1 + 2e-3) + 2e-3


def test_unit_activity_population_layers_and_refusals(tiny_session: Any) -> None:
    from drowse.errors import RequestError
    from drowse.model.units import UnitIndexError

    session = tiny_session
    node = _node(session, max_tokens=4)
    activity = session.unit_activity(node.id, top=3, raw_index=1, layers="0,2")
    population = cast(Any, activity["population"])
    assert [layer["layer"] for layer in population["layers"]] == [0, 2]
    assert activity["traces"] == {} and len(activity["top"]) == 3
    assert session.unit_activity(node.id, top=2)["population"] is None
    with pytest.raises(RequestError, match="raw_index"):
        session.unit_activity(node.id, layers="0")
    with pytest.raises(RequestError):
        session.unit_activity(node.id, top=0)
    with pytest.raises(UnitIndexError):
        session.unit_activity(node.id, units=["neuron/L1n9999"])
    with pytest.raises(RequestError, match="twice"):
        session.unit_activity(node.id, units=["neuron/L1n3", "neuron/L1n3"])


def test_a_pole_silent_on_the_neutral_corpus_reads_over_the_soft_floor(tiny_session: Any) -> None:
    """Regression (SmolLM2): poles whose neutral act_top is 0 read act / 1e-6
    (-554688 for neuron/L31n1094), and once those were cut, a pole with a tiny
    one (L31n2394+, 0.055 against its layer's 2.3) still topped the list at
    169.  Every value now reads over the soft scale, sqrt(s² + (c·m)²): a
    silent pole reads over c·m, finite and ranked with the rest, and its
    trace, its peak and its probe reading are one number."""
    from drowse.units.stats import SCALE_FLOOR_SHARE, typical_scale

    session = tiny_session
    node = _node(session, max_tokens=4)
    layer = session.units.runtime.stats().layer(1)
    layer.neuron_scale[:, 3] = 0.0
    layer.__dict__.pop("neuron_reading_scale", None)    # the cached soft scale, recomputed
    floor = SCALE_FLOOR_SHARE * typical_scale(layer.neuron_scale)
    torch.testing.assert_close(layer.neuron_reading_scale[:, 3], floor.expand(2))
    activity = session.unit_activity(node.id, top=1024, units=["neuron/L1n3"])
    trace = activity["traces"]["neuron/L1n3"]
    peak = next(p for p in activity["top"] if p["unit"] == "neuron/L1n3")
    assert abs(peak["peak"]) == pytest.approx(max(abs(v) for v in trace), abs=1e-5)
    assert all(abs(p["peak"]) < 1e3 for p in activity["top"])
    session.add_probe("neuron/L1n3")
    for raw_index in range(len(node.raw_token_ids)):
        replay = session.units.token_readout(node.id, raw_index)["measurements"]
        value = replay["instruments"]["units"]["readings"]["neuron/L1n3"]["value"]
        assert trace[raw_index] == pytest.approx(value, abs=1e-5)


def test_unit_activity_needs_unit_statistics(tiny_session: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from drowse.units.runtime import UnitsUnavailableError

    session = tiny_session
    node = _node(session, max_tokens=3)

    def unavailable() -> Any:
        raise UnitsUnavailableError("units are unavailable for this model: test")

    monkeypatch.setattr(session.units.runtime, "stats", unavailable)
    with pytest.raises(UnitsUnavailableError) as info:
        session.unit_activity(node.id)
    assert info.value.user_message()[0] == 422
