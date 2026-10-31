"""The S family in a session: ``slens/`` atoms, probes and gates, the
four-lens ``read``, and the ``tokens-in`` decomposition, on a tiny session
with a hand-built S lens (identity plus noise per layer)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from drowse import SamplingConfig
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
    session: Any, *, sites: tuple[str, ...] = ("out",), seed: int = 0,
) -> JacobianLens:
    """Publish a local S lens for the session's exact weights (``local:s``)."""
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.model.fingerprint import loaded_model_fingerprint

    n_layers = len(session.layers)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(seed)
    mats = {L: torch.eye(d) + 0.1 * torch.randn(d, d, generator=gen) for L in range(n_layers)}
    lens = JacobianLens(
        mats if "out" in sites else None, mid=mats if "mid" in sites else None,
        n_prompts=3, d_model=d, key=LensKey("S"),
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
    _install_s_lens(session, sites=("mid",))
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
    assert reading.value == pytest.approx(sum(expected.values()) / len(expected), rel=1e-4)
    depth = cast(Any, reading.depth)
    assert depth.basis == "footprint_z"
    mass = {L: max(z, 0.0) for L, z in expected.items()}
    n = len(session.layers) - 1
    center = sum(m * L / n for L, m in mass.items()) / sum(mass.values())
    assert depth.center == pytest.approx((center,), rel=1e-4)

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
    session.lens.fit(_TINY_PROMPTS)
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
