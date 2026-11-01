"""The loom-anchored token replay (:class:`drowse.loom.replay.TokenReplayer`).

A token replay rebuilds the forward that *produced* a loom token and reads it
with one instrument family.  The contract every family shares is that the
replay is the producing forward: on a generation that ran with live per-token
readouts, ``session.<family>.token_readout(node, raw_index)`` reports what the
live readout reported for that token (up to KV-cache-vs-full-forward rounding),
under the node's recorded steering.  ``steered=False`` reads the unsteered
counterfactual instead.  The first tests pin that end to end on a tiny
session; the rest pin the replayer's rebuild on a toy model.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, SamplingConfig
from drowse.generation.results import TokenEvent
from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
from drowse.loom.errors import InvalidNodeOperationError, UnknownNodeError
from drowse.loom.recipe import Recipe
from drowse.readout.measurements import token_row_measurements
from tests.support.replay import PROMPT_RENDER, ReplayRig
from tests.support.sessions import TinySessionFactory, register_vectors

_LENS_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "A small corpus for the tiny lens fit.",
    "Every row of the batch reads its own tail.",
]

def _live_consumer(events: list[TokenEvent]) -> TokenConsumer:
    return TokenConsumer(
        on_event=events.append,
        options=TokenConsumerOptions(
            live_scores=True, lens_readout=True, sae_readout=True,
        ),
    )

_STEERING = "0.4 mood"


def _all_families_session(
    factory: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """A scripted tiny session with a geometry probe, a fitted J-lens with a
    ``jlens/`` probe and a resident mock SAE with an ``sae/`` probe, every
    family's live readout on."""
    from drowse.sae import runtime as sae_runtime
    from drowse.sae import sources as sae_sources
    from drowse.sae.backend import ResolvedSae
    from tests.support.sae import MockSaeBackend

    session, _script = factory.scripted("Hi there, friend.")
    register_vectors(session, "mood")
    session.add_probe("mood")
    session.lens.fit(_LENS_CORPUS)
    width = int(session.model.config.hidden_size)
    backend = MockSaeBackend(layers=frozenset({1}), d_model=width)
    monkeypatch.setattr(
        sae_runtime, "resolve_sae_backend",
        lambda *_a, **_k: ResolvedSae(backend, 1, width),
    )
    monkeypatch.setattr(sae_sources, "load_sae_feature_meta", lambda *_a, **_k: {})
    monkeypatch.setattr(sae_sources, "save_sae_metadata", lambda *_a, **_k: None)
    session.sae.load("mock-release", layer=1)
    session.add_probe("jlens/a")
    session.add_probe("sae/1")
    session.lens.set_live(True)
    session.sae.set_live(True)
    return session


def _live_rows(session: Any) -> tuple[Any, dict[int, dict[str, Any]]]:
    """Each token's live envelope (the token frame's, every per-layer trace
    included), keyed by raw index; its loom row keeps the generated row's
    projection (the scores, each geometry share and coordinates)."""
    events: list[TokenEvent] = []
    session.generate(
        "Say hi.", steering=_STEERING,
        sampling=SamplingConfig(max_tokens=5), on_token=_live_consumer(events),
    )
    node = session.tree.get(session.tree.active_node_id)
    live: dict[int, Any] = {
        int(cast(int, event.raw_index)): event.measurements for event in events
    }
    assert len(live) == len(node.raw_token_ids) == 5
    for row in node.tokens:
        assert row["measurements"] == token_row_measurements(live[int(row["raw_index"])])
    return node, live


def _assert_scalar_reading(got: Any, want: Any) -> None:
    """A replayed lens or SAE probe reading is the live one: its value, its
    unit and its per-layer trace."""
    assert got["unit"] == want["unit"]
    assert got["value"] == pytest.approx(want["value"], rel=1e-4, abs=1e-7)
    assert set(got["per_layer"]) == set(want["per_layer"])
    for layer, value in want["per_layer"].items():
        assert got["per_layer"][layer] == pytest.approx(value, rel=1e-4, abs=1e-7)


def _replay(session: Any, family: str, node_id: str, k: int, **kw: Any) -> Any:
    envelope = session.instruments[family].token_readout(node_id, k, **kw)
    measurements = envelope["measurements"]
    assert measurements["scope"] == "replay"
    assert measurements["provenance"] == "replayed"
    return measurements["instruments"][family]


def test_every_family_replays_the_producing_forward(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, rows = _live_rows(session)

    for k, live in rows.items():
        live = live["instruments"]

        geometry = _replay(session, "geometry", node.id, k)
        assert geometry["binding"] == {"source": None, "steering": _STEERING}
        got, want = geometry["readings"]["mood"], live["geometry"]["readings"]["mood"]
        assert got["coords"] == pytest.approx(want["coords"], rel=1e-4, abs=1e-5)
        assert got["fraction"] == pytest.approx(want["fraction"], rel=1e-4, abs=1e-5)
        for layer, coords in want["coords_per_layer"].items():
            assert got["coords_per_layer"][layer] == pytest.approx(
                coords, rel=1e-4, abs=1e-5,
            )

        sae = _replay(session, "sae", node.id, k)
        assert sae["binding"] == live["sae"]["binding"]
        # The pinned probe replays with the features: a loom row keeps only
        # its value, and the token view draws its reading from the replay.
        _assert_scalar_reading(sae["readings"]["sae/1"], live["sae"]["readings"]["sae/1"])
        live_features = {f["id"]: f["activation"] for f in live["sae"]["readout"]["features"]}
        replay_features = {f["id"]: f["activation"] for f in sae["readout"]["features"]}
        assert set(replay_features) == set(live_features)
        for fid, activation in live_features.items():
            assert replay_features[fid] == pytest.approx(activation, rel=1e-4, abs=1e-6)

        # Replay wide enough that every live top token is in the replayed
        # set, then compare probabilities by token id (the tiny lens is
        # nearly flat, so the ranking itself can swap near-ties).
        lens = _replay(session, "lens", node.id, k, top_k=256)
        assert lens["binding"] == live["lens"]["binding"]
        _assert_scalar_reading(lens["readings"]["jlens/a"], live["lens"]["readings"]["jlens/a"])
        anatomy = session.token_anatomy(node.id, k, patterns=0)["measurements"]
        _assert_scalar_reading(
            anatomy["instruments"]["lens"]["readings"]["jlens/a"],
            live["lens"]["readings"]["jlens/a"],
        )
        replay_layers = {
            row["layer"]: {t["id"]: t["logprob"] for t in row["tokens"]}
            for row in lens["readout"]["layers"]
        }
        for row in live["lens"]["readout"]["layers"]:
            replayed = replay_layers[row["layer"]]
            for token in row["tokens"]:
                assert math.exp(replayed[token["id"]]) == pytest.approx(
                    math.exp(token["logprob"]), rel=1e-4, abs=1e-7,
                )


def test_unsteered_replay_reads_the_counterfactual(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, rows = _live_rows(session)
    k = 2
    live = rows[k]["instruments"]

    geometry = _replay(session, "geometry", node.id, k, apply_steering=False)
    assert geometry["binding"] == {"source": None, "steering": None}
    assert geometry["readings"]["mood"]["coords"] != pytest.approx(
        live["geometry"]["readings"]["mood"]["coords"], rel=1e-3,
    )
    for family, key in (("lens", "sources"), ("sae", "source")):
        replayed = _replay(session, family, node.id, k, apply_steering=False)
        assert replayed["binding"]["steering"] is None
        assert replayed["binding"][key] == live[family]["binding"][key]
    # Re-reading the steered replay afterwards still lands on the live
    # reading: the counterfactual left no steering behind.
    again = _replay(session, "geometry", node.id, k)
    assert again["readings"]["mood"]["coords"] == pytest.approx(
        live["geometry"]["readings"]["mood"]["coords"], rel=1e-4, abs=1e-5,
    )


# ---------------------------------------------------------------------------
# The replayer itself (toy model, real loom tree, recorded collaborators)
# ---------------------------------------------------------------------------


def _hidden_keys(hidden: dict[int, Any]) -> list[int]:
    return sorted(hidden)


def test_replay_rebuilds_the_producing_forward() -> None:
    rig = ReplayRig()
    raw_ids = rig.tokenizer.encode("abcdefg")
    node_id = rig.turn(
        raw_ids, Recipe(system_prompt="Saved pirate instruction"),
        role_label="guide",
    )

    token, read = rig.replay(
        node_id, 3, layers=[0, 2], read=_hidden_keys, label="test replay",
    )

    # Prompt + raw[:3], pooled at the final position: the forward that
    # PRODUCED the token, never including the token itself.
    assert rig.captures == [(rig.prompt_len + 3, rig.prompt_len + 2, (0, 2))]
    assert read == [0, 2]
    assert (token.node_id, token.raw_index) == (node_id, 3)
    assert token.token_id == raw_ids[3]
    assert token.token_text == rig.tokenizer.decode([raw_ids[3]])
    assert token.steering is None
    # Continue-mode rebuild: no text resend — the history walk to the
    # node's parent carries every prior turn; the gen header opens the
    # node's own seat under its stamped label.
    assert rig.prepare_calls == [{
        "input": None, "raw": False, "thinking": False,
        "system_prompt": "Saved pirate instruction",
        "parent_node_id": rig.tree.get(node_id).parent_id,
        "user_role": None, "assistant_role": "guide",
        "gen_seat": "assistant", "add_generation_prompt": True,
    }]


def test_replay_at_index_zero_reads_the_prompt_only() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abc"))
    rig.replay(node_id, 0, layers=[1], read=_hidden_keys, label="test replay")
    assert rig.captures == [(rig.prompt_len, rig.prompt_len - 1, (1,))]


def test_replay_of_a_user_seat_turn_opens_the_user_seat() -> None:
    rig = ReplayRig()
    node_id = rig.turn(
        rig.tokenizer.encode("abc"), seat="user", role_label="narrator",
    )
    rig.replay(node_id, 1, layers=[0], read=_hidden_keys, label="test replay")
    call = rig.prepare_calls[0]
    assert call["gen_seat"] == "user"
    assert call["user_role"] == "narrator"
    assert call["assistant_role"] is None
    # No recipe: the system prompt is inherited, not cleared.
    assert call["system_prompt"] is INHERIT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("supports", "recipe_thinking", "expected"),
    [
        (True, None, True),     # unset: the family default
        (False, None, False),
        (True, False, False),   # the recipe's choice
        (False, True, False),   # a family that cannot think never does
    ],
)
def test_replay_resolves_the_recipe_thinking(
    supports: bool, recipe_thinking: bool | None, expected: bool,
) -> None:
    rig = ReplayRig(supports_thinking=supports)
    node_id = rig.turn(rig.tokenizer.encode("abc"), Recipe(thinking=recipe_thinking))
    rig.replay(node_id, 1, layers=[0], read=_hidden_keys, label="test replay")
    assert rig.prepare_calls[0]["thinking"] is expected


def test_replay_raw_mode_renders_the_flat_buffer() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abcd"))
    rig.replay(
        node_id, 1, layers=[0], read=_hidden_keys, raw=True, label="test replay",
    )
    call = rig.prepare_calls[0]
    assert (call["input"], call["raw"]) == ("", True)
    assert call["parent_node_id"] == rig.tree.get(node_id).parent_id


def test_replay_reopens_the_recipe_steering_outside_the_exclusive_section() -> None:
    rig = ReplayRig()
    node_id = rig.turn(
        rig.tokenizer.encode("abcd"), Recipe(steering="0.3 formal.casual"),
    )

    def read(hidden: dict[int, Any]) -> None:
        rig.events.append("read")

    def layers() -> list[int]:
        rig.events.append("layers")
        return [0]

    token, _ = rig.replay(node_id, 2, layers=layers, read=read, label="test replay")
    assert token.steering == "0.3 formal.casual"
    assert rig.steering_scopes == ["0.3 formal.casual"]
    # The scope wraps the exclusive section (the composer's push/pop would
    # deadlock inside it) and so the render (the steering role shapes it).
    # The render uploads the ids to the device, so it runs inside the section
    # with the layer set and the read: outside it, it raced lock-free device
    # reads for MPS's one command buffer and aborted the server.
    assert rig.events == [
        "scope-open", "exclusive-enter", "render", "layers", "capture",
        "read", "exclusive-exit", "scope-close",
    ]
    assert rig.exclusive_messages == [(
        "test replay called while another model use is in flight",
        "test replay called while a generation is in flight",
    )]

    # The unsteered counterfactual opens no scope and reports no steering.
    rig.events.clear()
    token, _ = rig.replay(
        node_id, 2, layers=[0], read=read, apply_steering=False,
        label="test replay",
    )
    assert token.steering is None
    assert rig.steering_scopes == ["0.3 formal.casual"]
    assert "scope-open" not in rig.events


def test_replay_refuses_a_bad_anchor_before_any_work() -> None:
    rig = ReplayRig()
    node_id = rig.turn(rig.tokenizer.encode("abc"))
    user_id = rig.tree.get(node_id).parent_id
    assert user_id is not None and rig.tree.root_id is not None

    def replay(target: str, index: int) -> None:
        rig.replay(target, index, layers=[0], read=_hidden_keys, label="test replay")

    with pytest.raises(UnknownNodeError):
        replay("nope", 0)
    # An authored turn addresses its captured prompt tokens, and this one
    # no generation has measured yet.
    with pytest.raises(
        InvalidNodeOperationError, match="test replay: .*no generation has measured",
    ):
        replay(user_id, 0)
    # The system root stays out of bounds: not a turn at all.
    with pytest.raises(InvalidNodeOperationError, match="only a turn"):
        replay(rig.tree.root_id, 0)
    for index in (3, -1):
        with pytest.raises(InvalidNodeOperationError, match="out of range"):
            replay(node_id, index)
    # A finalized turn without raw ids (transcript import).
    bare = rig.tree.begin_assistant(user_id, recipe=Recipe())
    rig.tree.finalize_assistant(bare, text="no raw record", finish_reason="stop")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        replay(bare, 0)
    assert rig.events == []


def test_a_call_that_persists_the_traces_keeps_every_reading_whole(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated row keeps its scores and each geometry share and
    coordinates; a call that sets ``persist_per_layer_scores`` keeps every
    reading whole with its traces instead.  Either way the node keeps its
    reply's aggregate once (``aggregate_measurements``), and both survive the
    tree's own round trip."""
    from drowse.loom.tree import LoomTree
    from drowse.readout.measurements import aggregate_measurements, summary_measurements

    session = _all_families_session(make_tiny_session, monkeypatch)
    events: list[TokenEvent] = []
    result = session.generate(
        "Say hi.", steering=_STEERING, on_token=_live_consumer(events),
        sampling=SamplingConfig(max_tokens=3, persist_per_layer_scores=True),
    ).first
    node = session.tree.get(session.tree.active_node_id)
    live = {int(cast(int, event.raw_index)): event.measurements for event in events}
    for row in node.tokens:
        assert row["measurements"] == summary_measurements(
            cast(Any, live[int(row["raw_index"])]), per_layer=True,
        )
        assert "coords_per_layer" in row["measurements"]["instruments"]["geometry"]["readings"]["mood"]
    assert result.measurements is not None
    kept = node.aggregate_measurements
    assert kept == aggregate_measurements(cast(Any, result.measurements))
    assert kept is not None and set(kept["instruments"]) >= {"geometry", "lens", "sae"}
    again = LoomTree.from_dict(session.tree.to_dict(include_tokens=True))
    assert again.get(node.id).aggregate_measurements == kept
    assert again.get(node.id).tokens == node.tokens


# ---------------------------------------------------------------------------
# Authored prompt tokens (the node's PromptCapture anchor)
# ---------------------------------------------------------------------------


def _capture_whole(session: Any, text: str, *, steering: str | None) -> tuple[Any, Any]:
    """Generate after a fresh authored turn with every row kept whole
    (``persist_per_layer_scores``): the capture's own full record.  Returns
    the authored node and the reply."""
    events: list[TokenEvent] = []
    session.generate(
        text, steering=steering, on_token=_live_consumer(events),
        sampling=SamplingConfig(max_tokens=3, persist_per_layer_scores=True),
    )
    reply = session.tree.get(session.tree.active_node_id)
    assert reply.parent_id is not None
    return session.tree.get(reply.parent_id), reply


def _close(got: Any, want: Any, *, path: str = "") -> None:
    """``got`` is ``want`` up to replay rounding: numbers within 1e-4
    relative (1e-5 absolute), everything else equal."""
    if isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want, path
    elif isinstance(want, (int, float)):
        assert got == pytest.approx(want, rel=1e-4, abs=1e-5), path
    elif isinstance(want, dict):
        assert set(got) == set(want), path
        for key, value in want.items():
            _close(got[key], value, path=f"{path}.{key}")
    else:
        assert len(got) == len(want), path
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            _close(a, b, path=f"{path}[{i}]")


def _assert_replays_whole_rows(session: Any, node: Any, *, apply_steering: bool = True) -> None:
    """Every captured prompt row replays to its whole capture: each family's
    readings (summary and traces), the SAE features, the lens readout's
    layers (by vocabulary id) and the bindings."""
    from drowse.readout.measurements import summary_measurements

    rows = [*(node.thinking_tokens or ()), *(node.tokens or ())]
    measured = [k for k, row in enumerate(rows) if "measurements" in row]
    assert measured, "the capture measured the prompt"
    for k in measured:
        whole = rows[k]["measurements"]["instruments"]
        envelope: dict[str, Any] = {
            "version": 2, "scope": "replay", "provenance": "replayed", "instruments": {},
        }
        for family in ("geometry", "lens", "sae"):
            kw: dict[str, Any] = {"apply_steering": apply_steering}
            if family != "geometry":
                kw["top_k"] = 256 if family == "lens" else 32
            envelope["instruments"][family] = _replay(session, family, node.id, k, **kw)
        replayed: Any = summary_measurements(cast(Any, envelope), per_layer=True)["instruments"]
        _close(replayed["geometry"]["readings"], whole["geometry"]["readings"], path=f"[{k}].geometry")
        steering = node.prompt_capture.steering if apply_steering else None
        assert replayed["geometry"]["binding"] == {"source": None, "steering": steering}
        _close(replayed["sae"]["readings"], whole["sae"]["readings"], path=f"[{k}].sae")
        _close(replayed["lens"]["readings"], whole["lens"]["readings"], path=f"[{k}].lens")
        assert replayed["lens"]["binding"] == whole["lens"]["binding"]
        assert replayed["sae"]["binding"] == whole["sae"]["binding"]
        features = {f["id"]: f["activation"] for f in replayed["sae"]["readout"]["features"]}
        for feature in whole["sae"]["readout"]["features"]:
            assert features[feature["id"]] == pytest.approx(
                feature["activation"], rel=1e-4, abs=1e-6,
            )
        layers = {
            row["layer"]: {t["id"]: t["logprob"] for t in row["tokens"]}
            for row in replayed["lens"]["readout"]["layers"]
        }
        for row in whole["lens"]["readout"]["layers"]:
            for token in row["tokens"]:
                assert math.exp(layers[row["layer"]][token["id"]]) == pytest.approx(
                    math.exp(token["logprob"]), rel=1e-4, abs=1e-7,
                )


def test_an_authored_row_is_stored_slim_with_its_anchor(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: authored prompt rows were kept whole (every reading with
    its traces, the lens readout's layers, the SAE features: ~45 kB a token
    on SmolLM2) because no replay could reach them.  They keep what a
    generated row keeps, and the node carries the replay anchor."""
    from drowse.readout.measurements import ROW_GEOMETRY_KEYS

    session = _all_families_session(make_tiny_session, monkeypatch)
    events: list[TokenEvent] = []
    session.generate(
        "Say hi.", steering=_STEERING, on_token=_live_consumer(events),
        sampling=SamplingConfig(max_tokens=3),
    )
    reply = session.tree.get(session.tree.active_node_id)
    node = session.tree.get(cast(str, reply.parent_id))
    assert node.recipe is None and node.tokens
    capture = node.prompt_capture
    assert capture is not None and capture.steering == _STEERING
    assert capture.thinking_start is None and capture.start is not None
    for row in node.tokens:
        assert "raw_index" not in row
        measurements = row.get("measurements")
        if measurements is None:
            continue
        assert set(measurements["instruments"]) == {"geometry"}
        reading = measurements["instruments"]["geometry"]["readings"]["mood"]
        assert tuple(reading) == ROW_GEOMETRY_KEYS
        assert set(measurements["scores"]) == {"mood", "jlens/a", "sae/1"}


def test_an_authored_token_replays_its_whole_capture_steered(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, _reply = _capture_whole(session, "Say hi.", steering=_STEERING)
    assert node.prompt_capture is not None and node.prompt_capture.steering == _STEERING
    _assert_replays_whole_rows(session, node)
    # The counterfactual differs: the capture's prompt-phase steering is real.
    k = len(node.tokens) - 1
    unsteered = _replay(session, "geometry", node.id, k, apply_steering=False)
    assert unsteered["binding"]["steering"] is None
    captured = node.tokens[k]["measurements"]["instruments"]["geometry"]["readings"]["mood"]
    assert unsteered["readings"]["mood"]["coords"] != pytest.approx(captured["coords"], rel=1e-3)


def test_an_authored_token_replays_its_whole_capture_unsteered(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, _reply = _capture_whole(session, "Say hi.", steering=None)
    assert node.prompt_capture is not None and node.prompt_capture.steering is None
    _assert_replays_whole_rows(session, node)
    _assert_replays_whole_rows(session, node, apply_steering=False)


def test_an_authored_replay_survives_deleting_and_regenerating_other_nodes(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor lives on the authored node: deleting the reply that
    captured it, and regenerating under it with other steering, leaves its
    replay on the capture's own render and steering."""
    session = _all_families_session(make_tiny_session, monkeypatch)
    node, reply = _capture_whole(session, "Say hi.", steering=_STEERING)
    capture = node.prompt_capture
    session.generate(
        None, parent_node_id=node.id, steering="-0.4 mood",
        on_token=_live_consumer([]), sampling=SamplingConfig(max_tokens=2),
    )
    session.tree.delete_subtree(reply.id)
    assert reply.id not in session.tree.nodes
    assert session.tree.get(node.id).prompt_capture == capture  # one-shot
    _assert_replays_whole_rows(session, session.tree.get(node.id))


def test_an_edit_reads_the_original_and_an_edited_earlier_turn_refuses(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    first, _reply = _capture_whole(session, "Say hi.", steering=None)
    second, _ = _capture_whole(session, "And again, please.", steering=_STEERING)
    # Editing the node's own text keeps its rows and their render (as an
    # edited reply keeps its raw decode record).
    session.tree.edit(second.id, "Something else entirely.")
    _assert_replays_whole_rows(session, session.tree.get(second.id))
    # An earlier turn edited since: the capture's prompt no longer renders.
    session.tree.edit(first.id, "Say hello.")
    with pytest.raises(InvalidNodeOperationError, match="no longer renders the same"):
        _replay(session, "geometry", second.id, 0)


_UNITS = ("neuron/L1n3", "head/L2h1")


def _assert_activity_is_the_unit_probes(
    session: Any, node_id: str, *, apply_steering: bool = True,
) -> dict[str, list[float]]:
    """Unit activity over an authored turn reads every captured row as its
    own token replay reads the attached unit probes there; returns the
    traces."""
    activity = session.unit_activity(
        node_id, top=4, units=list(_UNITS), raw_index=1, apply_steering=apply_steering,
    )
    node = session.tree.get(node_id)
    rows = [*(node.thinking_tokens or ()), *(node.tokens or ())]
    assert activity["node_id"] == node_id
    assert [t["token_id"] for t in activity["tokens"]] == [r["token_id"] for r in rows]
    capture = node.prompt_capture
    assert activity["steering"] == (capture.steering if apply_steering else None)
    traces = activity["traces"]
    for k in range(len(rows)):
        replay = session.units.token_readout(node_id, k, apply_steering=apply_steering)
        readings = replay["measurements"]["instruments"]["units"]["readings"]
        for name in _UNITS:
            assert traces[name][k] == pytest.approx(readings[name]["value"], abs=1e-5), (name, k)
    population = cast(Any, activity["population"])
    neurons = {layer["layer"]: layer["neurons"] for layer in population["layers"]}
    from drowse.loom.node_replay import decode_population
    assert float(decode_population(neurons[1])[3]) == pytest.approx(
        traces["neuron/L1n3"][1], rel=2e-3, abs=2e-3,
    )
    return traces


def test_an_authored_turn_has_unit_activity_steered(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: unit activity refused an authored turn (400) once its
    prompt rows could replay; it reads them through the node's prompt
    capture in one prefill, as the token readouts do."""
    session = _all_families_session(make_tiny_session, monkeypatch)
    for name in _UNITS:
        session.add_probe(name)
    node, _reply = _capture_whole(session, "Say hi there, my friend.", steering=_STEERING)
    steered = _assert_activity_is_the_unit_probes(session, node.id)
    # The rows the capture scored live read the same values.
    for k, row in enumerate(node.tokens):
        scores = row.get("measurements", {}).get("scores", {})
        for name in _UNITS:
            if name in scores:
                assert steered[name][k] == pytest.approx(scores[name], abs=1e-4), (name, k)
    unsteered = _assert_activity_is_the_unit_probes(session, node.id, apply_steering=False)
    assert unsteered["neuron/L1n3"] != pytest.approx(steered["neuron/L1n3"], rel=1e-3)


def test_an_authored_turn_has_unit_activity_unsteered(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    for name in _UNITS:
        session.add_probe(name)
    node, _reply = _capture_whole(session, "Say hi there, my friend.", steering=None)
    plain = _assert_activity_is_the_unit_probes(session, node.id)
    assert plain == _assert_activity_is_the_unit_probes(session, node.id, apply_steering=False)


def test_authored_unit_activity_reads_each_channel_at_its_place(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two captured channels with a gap between them (a thinking channel,
    then the visible one): every row reads at its own producing position."""
    from drowse.loom.tree import PromptCapture, prompt_digest

    session = _all_families_session(make_tiny_session, monkeypatch)
    for name in _UNITS:
        session.add_probe(name)
    node, _reply = _capture_whole(session, "Say hi there, my friend.", steering=None)
    capture = node.prompt_capture
    assert capture is not None and capture.start is not None and len(node.tokens) >= 5
    twin = session.tree.add_user_turn(node.text, node.parent_id, dedup_existing=False)
    render = session.renderer.prepare_input(
        None, raw=capture.raw, thinking=capture.thinking, parent_node_id=twin,
        system_prompt=capture.system_prompt, add_generation_prompt=False,
    )[0].tolist()
    start = capture.start
    rows = [{"token_id": int(r["token_id"]), "text": r["text"]} for r in node.tokens]
    session.tree.set_authored_capture(
        twin,
        PromptCapture(
            steering=None, system_prompt=capture.system_prompt, thinking=capture.thinking,
            raw=capture.raw, start=start + 3, thinking_start=start,
            digest=prompt_digest(render[:start + 3]),
        ),
        thinking_tokens=rows[:2], tokens=rows[3:],
    )
    traces = _assert_activity_is_the_unit_probes(session, twin)
    whole = session.unit_activity(node.id, top=1, units=list(_UNITS))["traces"]
    for name in _UNITS:
        assert traces[name] == pytest.approx(whole[name][:2] + whole[name][3:], abs=1e-5)


def test_authored_unit_activity_refuses_an_edited_earlier_turn(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _all_families_session(make_tiny_session, monkeypatch)
    first, _reply = _capture_whole(session, "Say hi.", steering=None)
    second, _ = _capture_whole(session, "And again, please.", steering=_STEERING)
    session.tree.edit(second.id, "Something else entirely.")
    assert session.unit_activity(second.id, top=2)["tokens"]
    session.tree.edit(first.id, "Say hello.")
    with pytest.raises(InvalidNodeOperationError, match="no longer renders the same"):
        session.unit_activity(second.id, top=2)


def test_an_authored_forward_places_each_channel() -> None:
    rig = ReplayRig()
    user_id, _render = _rig_authored(rig)
    _token, forward = rig.replay.forward(user_id, 4, run=lambda f: f, label="t")
    # Thinking rows 0-1 at render positions 5-6, visible rows 2-4 at 9-11.
    assert forward.spans() == ((0, 2, 4), (2, 5, 8))
    assert [forward.producing_position(k) for k in range(5)] == [4, 5, 8, 9, 10]


def _rig_authored(
    rig: ReplayRig, *, steering: str | None = "0.3 mood", digest: str | None = None,
) -> tuple[str, list[int]]:
    """An authored turn with both channels captured inside the rig's render:
    thinking rows at render positions 5-6, visible rows at 9-11."""
    from drowse.loom.tree import PromptCapture, prompt_digest

    render = rig.tokenizer.encode(PROMPT_RENDER)
    user_id = rig.tree.add_user_turn("a user turn", thinking_text="hm")
    rig.tree.set_authored_capture(
        user_id,
        PromptCapture(
            steering=steering, system_prompt="be brief", thinking=True, raw=False,
            start=9, thinking_start=5,
            digest=prompt_digest(render[:9]) if digest is None else digest,
        ),
        thinking_tokens=[{"token_id": t, "text": "x"} for t in render[5:7]],
        tokens=[{"token_id": t, "text": "y"} for t in render[9:12]],
    )
    return user_id, render


def test_an_authored_index_runs_thinking_then_visible_through_one_prefill() -> None:
    rig = ReplayRig()
    user_id, render = _rig_authored(rig)
    for k in range(5):
        rig.replay(user_id, k, layers=[0], read=_hidden_keys, label="test replay")
    # Thinking rows 0-1 at positions 5-6, visible rows 2-4 at 9-11: each
    # replay is the render before the token, pooled at its producer.
    assert rig.captures == [(n, n - 1, (0,)) for n in (5, 6, 9, 10, 11)]
    assert rig.steering_scopes == ["0.3 mood"] * 5
    assert rig.prepare_calls[0] == {
        "input": None, "raw": False, "thinking": True, "system_prompt": "be brief",
        "parent_node_id": user_id, "user_role": None, "assistant_role": None,
        "gen_seat": "assistant", "add_generation_prompt": False,
    }
    token, _ = rig.replay(user_id, 3, layers=[0], read=_hidden_keys, label="t")
    assert token.raw_index == 3 and token.token_id == render[10]
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        rig.replay(user_id, 5, layers=[0], read=_hidden_keys, label="test replay")


def test_an_authored_replay_refuses_a_prompt_that_renders_otherwise() -> None:
    from drowse.loom.tree import prompt_digest

    rig = ReplayRig()
    user_id, _render = _rig_authored(rig, digest=prompt_digest([1, 2, 3]))
    with pytest.raises(InvalidNodeOperationError, match="no longer renders the same"):
        rig.replay(user_id, 0, layers=[0], read=_hidden_keys, label="test replay")
    assert rig.captures == []


def test_a_prompt_only_replay_fires_gated_and_phased_terms_as_the_prefill_did() -> None:
    """A prompt token is a prefill position: a probe gate stays shut and a
    phase window reads its prompt slot there (``Trigger.active``), so a
    replay with no decode position refuses neither, gate scorer or not."""
    from drowse.loom.replay import ReplaySegment, replay_segments
    from drowse.steering.triggers import Trigger

    for trigger in (
        Trigger.when("x", ">", 0.1), Trigger.AFTER_THINKING, Trigger.THINKING_ONLY,
    ):
        assert replay_segments(7, 7, [trigger], can_think=True) == (
            ReplaySegment(0, 7, True, 0),
        )
