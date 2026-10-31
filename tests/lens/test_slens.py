"""The S family in a session: ``slens/`` atoms, probes and gates, the
four-lens ``read``, and the ``tokens-in`` decomposition, on a tiny session
with a hand-built S lens (identity plus noise per layer)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from drowse import SamplingConfig
from drowse.lens.depth import LensBand, kind_weights
from drowse.lens.footprint import token_mask
from drowse.lens.jlens import JacobianLens, LensNotFittedError
from drowse.lens.key import LensKey
from drowse.lens.read import LensRead
from tests.support.lens import FakeCapture, LensHarness
from tests.support.sessions import forward_logits

pytestmark = pytest.mark.usefixtures("isolated_home")

_TINY_PROMPTS = [
    "a first prompt that is long enough for the tiny model",
    "the second prompt, also long enough for the tiny one",
    "and a third one to round out the tiny model's corpus",
]


def _install_s_lens(
    session: Any, *, sites: tuple[str, ...] = ("out",), seed: int = 0, kind: str = "S",
) -> JacobianLens:
    """Publish (and select) a local lens of ``kind`` for the session's exact
    weights: ``local:s`` by default."""
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.model.fingerprint import loaded_model_fingerprint

    n_layers = len(session.layers)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(seed)
    mats = {L: torch.eye(d) + 0.1 * torch.randn(d, d, generator=gen) for L in range(n_layers)}
    lens = JacobianLens(
        mats if "out" in sites else None, mid=mats if "mid" in sites else None,
        n_prompts=3, d_model=d, key=LensKey(kind),
    )
    save_lens(lens, session.model_id, LensProvenance(
        corpus_spec="test", corpus_sha256="a" * 64, seq_len=8, dim_batch=1,
        model_fingerprint=loaded_model_fingerprint(session.model, session.model_id),
    ))
    return lens


def _centred(session: Any, token_id: int) -> torch.Tensor:
    """``ẽ_v`` by hand: the llama tiny model's embedding has scale 1."""
    weight = session.model.get_input_embeddings().weight.detach().float()
    keep = token_mask(session.tokenizer, int(weight.shape[0]))
    return weight[token_id] - weight[keep].mean(0)


def _z(session: Any, lens: JacobianLens, layer: int, h: torch.Tensor, token_id: int) -> float:
    weight = session.model.get_input_embeddings().weight.detach().float()
    keep = token_mask(session.tokenizer, int(weight.shape[0]))
    centred = weight[keep] - weight[keep].mean(0)
    u = h.float() @ lens.matrix(layer, "out")
    scores = centred @ u
    return float((u @ _centred(session, token_id)) / scores.std(correction=0))


def _token(session: Any, word: str) -> int:
    from drowse.lens.jlens import resolve_word_token

    return resolve_word_token(session.tokenizer, word)


# -- steering atoms -----------------------------------------------------------


def test_slens_atom_is_the_footprint_and_steers(make_tiny_session: Any) -> None:
    session = make_tiny_session()
    lens = _install_s_lens(session)
    a = _token(session, "a")
    profile = session.registry.ensure_profile_registered("slens/a")
    assert sorted(profile) == list(range(len(session.layers)))
    for layer, vector in profile.items():
        expected = lens.matrix(layer, "out") @ _centred(session, a)
        assert torch.allclose(vector.cpu(), expected, atol=1e-5)
    by_id = session.registry.ensure_profile_registered(f"slens/#{a}")
    assert all(torch.allclose(by_id[L], profile[L]) for L in profile)

    base = forward_logits(session)
    with session.steering("0.5 slens/a"):
        full = forward_logits(session)
    with session.steering("0.5 slens/a{1-2}"):
        scoped = forward_logits(session)
    assert not torch.allclose(base, full)
    assert not torch.allclose(base, scoped) and not torch.allclose(full, scoped)


def test_slens_atom_needs_an_s_lens_with_an_out_site(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    with pytest.raises(LensNotFittedError, match="rs-s"):
        session.registry.ensure_profile_registered("slens/a")
    _install_s_lens(session, sites=("mid",))  # saved unselected (lens_selectable)
    session.lens.select_source("local:s")
    # A mid-only S lens never stands in for out: steering injects at block outputs.
    with pytest.raises(LensNotFittedError, match="no 'out' site"):
        session.registry.ensure_profile_registered("slens/a")
    with pytest.raises(LensNotFittedError, match="no 'out' site"):
        session.add_probe("slens/a")


def test_switching_the_s_source_drops_slens_directions_only(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    session.lens.fit(_TINY_PROMPTS)
    _install_s_lens(session)
    session.registry.ensure_profile_registered("jlens/a")
    session.registry.ensure_profile_registered("slens/a")
    _install_s_lens(session, seed=1)  # a re-publication: a new identity
    assert session.lens.resident_input() is not None
    assert "slens/a" not in session.registry.profiles
    assert "jlens/a" in session.registry.profiles
    session.lens.select_source("local:j")
    assert "jlens/a" not in session.registry.profiles


def test_a_fit_drives_its_own_familys_residency_only(make_tiny_session: Any) -> None:
    """Regression: the fitter evicted and adopted only the R family, so an S
    or K fit left the S slot holding the replaced lens (and its ``slens/``
    directions) until a later disk refresh.  Each fit now evicts and adopts
    in its own family's slot and never touches the other's; a fit without
    ``out`` touches neither."""
    from drowse.base.source_registry import LOCAL, SourceRef
    from drowse.lens.sources import LENS_SOURCES
    from drowse.lens.store import lens_identity, load_local_lens_sidecar

    session = make_tiny_session(warm=False)
    lens = session.lens
    lens.fit(_TINY_PROMPTS, kind="R", sites=("out",))
    r_resident, r_identity = lens.resident_lens, lens.resident_identity
    _install_s_lens(session, seed=3)  # hand-built local:s, selected
    assert lens.resident_input() is not None
    session.registry.ensure_profile_registered("jlens/a")
    session.registry.ensure_profile_registered("slens/a")
    session.add_probe("slens/a")
    lens.probes["slens/a"]["layers"] = []

    lens.fit(_TINY_PROMPTS, kind="S", force=True)
    sidecar = load_local_lens_sidecar(session.model_id, "s")
    assert sidecar is not None
    # Adopted by the fit itself, without a resident_input() refresh.
    assert lens.input_runtime.identity == lens_identity(sidecar)
    assert lens.input_runtime.lens.sites == ("out", "mid")
    assert lens.probes["slens/a"]["layers"] == list(range(len(session.layers)))
    assert "slens/a" not in session.registry.profiles
    # The R family is untouched.
    assert lens.resident_lens is r_resident and lens.resident_identity == r_identity
    assert "jlens/a" in session.registry.profiles

    s_resident, s_identity = lens.input_runtime.lens, lens.input_runtime.identity
    session.registry.ensure_profile_registered("slens/a")
    lens.fit(_TINY_PROMPTS, kind="R", force=True)
    assert lens.resident_identity != r_identity
    assert lens.input_runtime.lens is s_resident and lens.input_runtime.identity == s_identity
    assert "slens/a" in session.registry.profiles

    # A fit without out selects and adopts nothing in either family.
    lens.fit(_TINY_PROMPTS, kind="K", sites=("mid",))
    assert LENS_SOURCES.active(session.model_id, "input") == SourceRef(LOCAL, "s")
    assert lens.input_runtime.lens is s_resident
    assert "slens/a" in session.registry.profiles


# -- probes and gates -----------------------------------------------------------


def test_slens_probe_reads_footprint_z(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    lens = _install_s_lens(session)
    a = _token(session, "a")
    assert session.add_probe("slens/a") == "slens/a"
    spec = session.lens.probes["slens/a"]
    assert spec == {
        "word": "a", "token_id": a, "layers": list(range(len(session.layers))), "side": "input",
    }
    gen = torch.Generator().manual_seed(4)
    d = lens.d_model
    hidden = {L: torch.randn(d, generator=gen) for L in spec["layers"]}
    reading = session.lens.score_probes(hidden)["slens/a"]
    assert reading.unit == "mean_footprint_z"
    expected = {L: _z(session, lens, L, hidden[L], a) for L in hidden}
    assert reading.per_layer == pytest.approx(expected, rel=1e-4)
    # Regression: the value was a plain mean over every S:out layer; it is
    # the depth rule's w_S-weighted mean, so it reads where S is faithful.
    n = len(session.layers) - 1
    w_s = {L: w for L, (w, _w_r) in enumerate(kind_weights(n + 1, session.lens.band))}
    weighted = sum(w_s[L] * z for L, z in expected.items()) / sum(w_s.values())
    assert reading.value == pytest.approx(weighted, rel=1e-4)
    assert reading.value != pytest.approx(sum(expected.values()) / len(expected), rel=1e-3)
    depth = cast(Any, reading.depth)
    assert depth.basis == "footprint_z"
    mass = {L: w_s[L] * max(z, 0.0) for L, z in expected.items()}
    center = sum(m * L / n for L, m in mass.items()) / sum(mass.values())
    assert depth.center == pytest.approx((center,), rel=1e-4)
    # The band moves the weights; a band with no S weight at any read layer
    # reads 0 and keeps every per-layer z.
    session.lens.band = LensBand(0.0, 0.0)
    flat = session.lens.score_probes({L: h for L, h in hidden.items() if L > 0})["slens/a"]
    assert flat.value == 0.0
    assert flat.per_layer == pytest.approx({L: z for L, z in expected.items() if L > 0}, rel=1e-4)
    session.lens.band = LensBand()

    # Its own probe identity: the S lens, never the R one.
    session.lens.fit(_TINY_PROMPTS)
    session.add_probe("jlens/a")
    hashes = session.lens.probe_hash("slens/a"), session.lens.probe_hash("jlens/a")
    assert None not in hashes and hashes[0] != hashes[1]
    both = session.lens.score_probes(hidden)
    assert both["slens/a"].unit == "mean_footprint_z"
    assert both["jlens/a"].unit == "mean_token_probability"


def test_slens_gate_scalars_and_the_step_stash(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    _install_s_lens(session)
    harness = LensHarness(
        model=session.model, tokenizer=session.tokenizer, layers=session.layers,
        model_id=session.model_id,
    )
    harness.lens.attach("slens/a")
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(5)
    harness.capture = FakeCapture({L: torch.randn(d, generator=gen) for L in range(len(session.layers))})
    calls: list[int] = []
    original = harness.lens._input_readings

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    harness.lens._input_readings = counting  # type: ignore[method-assign]
    scalars = harness.lens.gate_scalars({"slens/a"}, step_id=3)
    assert set(scalars) == {"slens/a", "slens/a[0]"}
    assert scalars["slens/a"] == scalars["slens/a[0]"]
    assert harness.lens.current_run.input_stash is not None
    assert harness.lens.current_run.input_stash["step"] == 3
    # The same forward reuses the stash; the next one reads again.
    assert harness.lens.gate_scalars({"slens/a"}, step_id=3) == scalars
    assert len(calls) == 1
    harness.lens.gate_scalars({"slens/a"}, step_id=4)
    assert len(calls) == 2
    assert harness.lens.gate_scalars(set(), step_id=5) == {}


def test_slens_gate_and_aggregate_through_a_generation(make_tiny_session: Any) -> None:
    session = make_tiny_session()
    _install_s_lens(session)
    session.add_probe("slens/a")
    with session.steering("0.3 slens/a@when:slens/a>3"):
        assert session._read_plane.resolve_gates().lens_keys == {"slens/a"}
    result = session.generate(
        "Say hi.", steering="0.3 slens/a@when:slens/a>3",
        sampling=SamplingConfig(max_tokens=3), stateless=True, thinking=False,
    ).first
    lens = cast(dict[str, Any], result.measurements)["instruments"]["lens"]
    reading = lens["readings"]["slens/a"]
    assert reading["unit"] == "mean_footprint_z"
    assert reading["depth"]["basis"] == "footprint_z"
    assert sorted(int(L) for L in reading["per_layer"]) == list(range(len(session.layers)))


def test_slens_gate_refuses_other_channels(make_tiny_session: Any) -> None:
    from drowse.errors import UnsupportedProbeChannelError

    session = make_tiny_session()
    _install_s_lens(session)
    session.add_probe("slens/a")
    with pytest.raises(UnsupportedProbeChannelError):
        session.generate(
            "Say hi.", steering="0.3 slens/a@when:slens/a:fraction>0.5",
            sampling=SamplingConfig(max_tokens=1), stateless=True, thinking=False,
        )


# -- the four-lens read -----------------------------------------------------------


def test_read_returns_both_families_by_depth(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    session.lens.fit(_TINY_PROMPTS, sites=("out",))
    s_lens = _install_s_lens(session, sites=("out", "mid"))
    d = s_lens.d_model
    vector = torch.randn(d, generator=torch.Generator().manual_seed(6))

    read = cast(LensRead, session.lens.read(vector, 0, "out", k=3))
    assert (read.layer, read.site, read.depth) == (0, "out", 0.0)
    assert (read.s.weight, read.r.weight) == (1.0, 0.0)
    assert (read.r.lens, read.s.lens) == ("J:out", "S:out")
    assert read.r.present and read.s.present
    assert len(read.r.tokens) == len(read.s.tokens) == 3
    assert read.s.tokens[0].value >= read.s.tokens[-1].value
    top = read.s.tokens[0]
    assert top.value == pytest.approx(_z(session, s_lens, 0, vector, top.token_id), rel=1e-4)

    deep = cast(LensRead, session.lens.read(vector, len(session.layers) - 1, "mid"))
    assert (deep.s.weight, deep.r.weight) == (0.0, 1.0)
    # The R lens holds out only: its mid half is absent, never read at out.
    assert not deep.r.present and "no 'mid' site" in str(deep.r.absent)
    assert deep.s.present and deep.s.lens == "S:mid"

    rows = session.lens.read(torch.stack([vector, -vector]), 1, "out", k=2)
    assert isinstance(rows, list) and len(rows) == 2
    from drowse.errors import RequestError

    with pytest.raises(RequestError, match="site"):
        session.lens.read(vector, 0, "pre")
    with pytest.raises(RequestError, match="outside"):
        session.lens.read(vector, len(session.layers), "out")


def test_read_without_lenses_is_two_absent_halves(make_tiny_session: Any) -> None:
    session = make_tiny_session(warm=False)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    read = cast(LensRead, session.lens.read(torch.ones(d), 1))
    assert not read.r.present and not read.s.present
    assert "rs-s" in str(read.s.absent) and read.r.lens is None


def test_read_prompt_reads_out_and_mid_residuals(make_tiny_session: Any) -> None:
    from drowse.model.blocks import model_spec
    from tests.support.blocks import spec_forward

    session = make_tiny_session(warm=False)
    _install_s_lens(session, sites=("out", "mid"))
    prompt = "abc def"
    ids = session.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    with torch.no_grad():
        mids, outs, _ = spec_forward(model_spec(session.model), ids)
    for site, rows in (("out", outs), ("mid", mids)):
        by_position = session.lens.read_prompt(prompt, site=site, positions=[-1, 1], k=2)
        assert sorted(by_position) == [1, len(ids) - 1]
        for position, reads in by_position.items():
            assert [read.layer for read in reads] == list(range(len(session.layers)))
            for read in reads:
                direct = cast(LensRead, session.lens.read(rows[read.layer][position], read.layer, site, k=2))
                assert [t.token_id for t in read.s.tokens] == [t.token_id for t in direct.s.tokens]
                assert [t.value for t in read.s.tokens] == pytest.approx(
                    [t.value for t in direct.s.tokens], rel=1e-3, abs=1e-4,
                )


# -- tokens-in decomposition ---------------------------------------------------------


def test_tokens_in_decomposition_finds_the_footprint(make_tiny_session: Any) -> None:
    from drowse.errors import RequestError

    session = make_tiny_session(warm=False)
    _install_s_lens(session)
    a = _token(session, "a")
    session.registry.ensure_profile_registered("slens/a")
    out = session.lens.decompose("slens/a", k=2, dictionary="tokens-in")
    assert sorted(out) == list(range(len(session.layers)))
    for share, tokens in out.values():
        assert share == pytest.approx(1.0, abs=1e-3)
        assert tokens[0][0] == session.tokenizer.decode([a])
    with pytest.raises(RequestError, match="dictionary"):
        session.lens.decompose("slens/a", dictionary="neurons")
    # tokens-out still reads the R family, which this session lacks.
    with pytest.raises(LensNotFittedError):
        session.lens.decompose("slens/a", dictionary="tokens-out")


# -- review regressions -----------------------------------------------------------


def _harness(session: Any) -> LensHarness:
    """A harness over the session's model and home, with a capture of every
    layer's slice."""
    harness = LensHarness(
        model=session.model, tokenizer=session.tokenizer, layers=session.layers,
        model_id=session.model_id,
    )
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(11)
    harness.capture = FakeCapture({L: torch.randn(d, generator=gen) for L in range(len(session.layers))})
    return harness


def _counting_input_reads(harness: LensHarness) -> list[set[str] | None]:
    calls: list[set[str] | None] = []
    original = harness.lens._input_readings

    def counting(*args: Any, only: Any = None, **kwargs: Any) -> Any:
        calls.append(None if only is None else set(only))
        return original(*args, only=only, **kwargs)

    harness.lens._input_readings = counting  # type: ignore[method-assign]
    return calls


def _count_sidecar_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count lens sidecar reads (the per-refresh disk hit) by family slot."""
    from drowse.lens import store

    reads: dict[str, int] = {}
    original = store.load_lens_sidecar

    def counting(model_id: str, side: str = "output") -> Any:
        reads[side] = reads.get(side, 0) + 1
        return original(model_id, side)

    monkeypatch.setattr(store, "load_lens_sidecar", counting)
    return reads


def test_a_gate_on_one_slens_probe_keeps_the_others_in_the_step_readings(
    make_tiny_session: Any,
) -> None:
    """Regression: the gate stashed readings for the gated subset only, and the
    display step on the same forward reused that partial stash, dropping every
    ungated ``slens/`` probe; a ``jlens/``-only gate emptied them all."""
    session = make_tiny_session(warm=False)
    _install_s_lens(session)
    harness = _harness(session)
    harness.lens.attach("slens/a")
    harness.lens.attach("slens/b")
    calls = _counting_input_reads(harness)
    gated = harness.lens.gate_scalars({"slens/a"}, step_id=3)
    assert set(gated) == {"slens/a", "slens/a[0]"}
    shown = harness.lens._input_step_readings(harness.lens.probes, None, step_id=3)
    assert sorted(shown) == ["slens/a", "slens/b"]
    # The display read only what the gate had not: b, once.
    assert calls == [{"slens/a"}, {"slens/b"}]
    assert harness.lens._input_step_readings(harness.lens.probes, None, step_id=3) == shown
    assert len(calls) == 2

    # A gate naming only a jlens/ probe reads no slens/ probe at all.
    harness.lens.probes["jlens/x"] = {"word": "x", "token_id": 1, "layers": [], "side": "output"}
    calls.clear()
    assert harness.lens.gate_scalars({"jlens/x"}, step_id=4) == {}
    assert calls == []
    shown = harness.lens._input_step_readings(harness.lens.probes, None, step_id=4)
    assert sorted(shown) == ["slens/a", "slens/b"]


def test_an_slens_only_gate_never_refreshes_the_r_lens_per_token(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with an R lens selected, live off and only ``slens/``
    probes gated, the gate callback resolved the unpinned R family every
    decode step (an R sidecar read per token)."""
    session, _script = make_tiny_session.scripted("abcdefghijklmnop")
    session.lens.fit(_TINY_PROMPTS)
    _install_s_lens(session)
    session.add_probe("slens/a")
    reads = _count_sidecar_reads(monkeypatch)

    def run(tokens: int) -> dict[str, int]:
        reads.clear()
        result = session.generate(
            "Say hi.", steering="0.3 slens/a@when:slens/a>-100",
            sampling=SamplingConfig(max_tokens=tokens), stateless=True, thinking=False,
        ).first
        assert result.token_count == tokens
        return dict(reads)

    run(2)  # the first run also resolves the slens/a atom
    short, long = run(2), run(8)
    assert short == long  # nothing scales with the token count
    assert long.get("output", 0) <= 1


def test_live_readout_pins_and_captures_the_s_family(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with live readout on and no aggregate or gate, ``slens/``
    probes read an unpinned S lens (refreshed from disk every step) over only
    the R live layers the capture happened to hold."""
    from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
    from drowse.readout.types import ReadRequest

    session, _script = make_tiny_session.scripted("abcdefghijklmnop")
    session.lens.fit(_TINY_PROMPTS)
    session.lens.set_live(True, layers=[0])
    _install_s_lens(session)
    session.add_probe("slens/a")
    every = list(range(len(session.layers)))

    prep = session.lens.prepare(ReadRequest(live=True, final_aggregate=False))
    assert prep.input_pinned and prep.input_lens is session.lens.resident_input()
    assert set(every) <= set(session.lens.plan(prep).latest_layers)

    reads = _count_sidecar_reads(monkeypatch)

    def run(tokens: int) -> tuple[dict[str, int], list[Any]]:
        reads.clear()
        events: list[Any] = []
        session.generate(
            "Say hi.", sampling=SamplingConfig(max_tokens=tokens, return_probe_readings=False),
            stateless=True, thinking=False,
            on_token=TokenConsumer(events.append, TokenConsumerOptions(lens_readout=True)),
        )
        return dict(reads), events

    run(2)
    short, _events = run(2)
    long, events = run(6)
    assert short == long  # the S lens is pinned at prepare, not read per token
    per_step = [
        e.measurements["instruments"]["lens"]["readings"]["slens/a"]
        for e in events if e.measurements and "lens" in e.measurements["instruments"]
    ]
    assert len(per_step) == 6
    assert all(sorted(int(L) for L in r["per_layer"]) == every for r in per_step)


def test_the_live_step_folds_slens_readings_into_its_one_transfer(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The display step reads ``slens/`` probes it has not stashed in the same
    packed host transfer as the readout, never in a second one."""
    import drowse.lens.instrument as instrument

    session = make_tiny_session(warm=False)
    session.lens.fit(_TINY_PROMPTS)
    _install_s_lens(session)
    harness = _harness(session)
    harness.lens.attach("slens/a")
    harness.lens.enable_live()
    expected = harness.lens.score_probes(harness.capture.latest_per_layer())["slens/a"]
    packs: list[int] = []
    original = instrument.pack_readout_rows_to_host

    def counting(*rows: torch.Tensor) -> torch.Tensor:
        packs.append(len(rows))
        return original(*rows)

    def separate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a second host transfer for the slens/ readings")

    monkeypatch.setattr(instrument, "pack_readout_rows_to_host", counting)
    harness.lens._input_readings = separate  # type: ignore[method-assign]
    step = harness.lens.live_readout_step(top_k=3, step_id=7)
    assert step is not None and len(packs) == 1
    reading = cast(Any, harness.lens.last_step_readings)["slens/a"]
    assert reading.value == pytest.approx(expected.value, rel=1e-5)
    assert reading.per_layer == pytest.approx(expected.per_layer, rel=1e-5)
    # Same forward again: the stash answers, nothing is read.
    assert harness.lens.live_readout_step(top_k=3, step_id=7) is not None
    assert cast(Any, harness.lens.last_step_readings)["slens/a"] == reading


def test_a_mid_only_r_source_leaves_jlens_probes_layerless_and_generation_runs(
    make_tiny_session: Any,
) -> None:
    """Regression: ``prepare`` probed ``source_layers``, which raises for a lens
    without ``out``, so a mid-only R source failed every generation that
    pinned the R family."""
    session = make_tiny_session()
    session.lens.fit(_TINY_PROMPTS)
    session.add_probe("jlens/a")
    _install_s_lens(session, sites=("mid",), kind="R")  # local:r, saved unselected
    session.lens.select_source("local:r")
    assert session.lens.resident().sites == ("mid",)
    assert session.lens.probes["jlens/a"]["layers"] == []
    result = session.generate(
        "Say hi.", sampling=SamplingConfig(max_tokens=2), stateless=True, thinking=False,
    ).first
    assert result.token_count >= 1


def test_slens_atom_resolution_runs_in_the_exclusive_section(make_tiny_session: Any) -> None:
    """Regression: the first ``slens/`` atom builds the embedding statistics
    (model work) and ran outside the model-exclusive section."""
    session = make_tiny_session(warm=False)
    lens = _install_s_lens(session)
    harness = _harness(session)
    assert harness.lens.register_input_direction("a") == "slens/a"
    assert len(harness.exclusive_messages) == 1
    assert "slens/" in harness.exclusive_messages[0]
    profile = harness.profiles["slens/a"]
    expected = lens.matrix(1, "out") @ _centred(session, _token(session, "a"))
    assert torch.allclose(profile[1].cpu(), expected, atol=1e-5)
    # Idempotent: a registered atom does no model work.
    harness.lens.register_input_direction("a")
    assert len(harness.exclusive_messages) == 1


def test_slens_refuses_a_token_the_mask_drops(make_tiny_session: Any) -> None:
    """Regression: ``slens/#<id>`` accepted special tokens and padding rows,
    which the S read never ranks, as atoms and probes."""
    from drowse.lens.jlens import InvalidTokenIdError

    session = make_tiny_session(warm=False)
    _install_s_lens(session)
    rows = int(session.model.get_input_embeddings().weight.shape[0])
    masked = int((~token_mask(session.tokenizer, rows)).nonzero()[0])
    with pytest.raises(InvalidTokenIdError, match="mask"):
        session.registry.ensure_profile_registered(f"slens/#{masked}")
    with pytest.raises(InvalidTokenIdError, match="mask"):
        session.add_probe(f"slens/#{masked}")
    assert f"slens/#{masked}" not in session.lens.probes


def test_read_refuses_a_vector_of_the_wrong_width(make_tiny_session: Any) -> None:
    """Regression: a wrong-width vector escaped as a raw torch error (a
    scrubbed 500 over a route) instead of a request error."""
    from drowse.errors import RequestError

    session = make_tiny_session(warm=False)
    _install_s_lens(session)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    for bad in (torch.ones(d - 1), torch.ones(2, d + 1)):
        with pytest.raises(RequestError, match="residual-width"):
            session.lens.read(bad, 0, "out")
