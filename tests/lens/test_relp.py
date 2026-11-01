"""CPU tests for the RelP (R-lens) rules and the R/J name and policy split.

``lens/relp.py`` names the rules (``RELP_RULES``); the transport estimator
applies them per block (``rules="relp"``, checked map by map against the
autograd oracle in ``test_lens_transport.py``).  Here: the rules as a
derivative (the transport's MLP and norm maps against the oracle's patched
autograd tangent), the oracle's patches themselves (forward values kept,
modules restored), and the ``session.lens.fit`` name/policy separation
(``local:r`` beside ``local:j``) through checkpoint, resume, promotion and
the no-op proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.cli.verbs.lens import lens_fit_is_noop
from drowse.lens.jlens import JacobianLens
from drowse.lens.relp import HALF_RULE_BETA, RELP_RULES
from drowse.base.source_registry import SourceRef
from drowse.lens.key import LensKey
from drowse.lens.store import (
    LensProvenance,
    lens_estimator_policy,
    lens_manifest_path,
    load_lens_checkpoint_sidecar,
    load_local_lens,
    load_local_lens_sidecar,
    remove_lens,
    save_lens,
    save_lens_checkpoint,
)
from drowse.lens.sources import LENS_SOURCES, list_lens_sources, use_lens_source
from drowse.lens.transport import linearize
from drowse.model.blocks import model_spec
from tests.support.blocks import tiny_dense_model
from tests.support.jlens_toys import TOY_D
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import LensHarness
from tests.support.lens_oracle import RelpOracleError, relp_backward_rules

_DENSE_MODEL_ID = "toy/relp-dense-model"
DENSE_D = TOY_D


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def _DenseLensHarness(*, n_layers: int = 3) -> LensHarness:
    """A lens harness on the toy llama, under its own model id."""
    return LensHarness(n_layers=n_layers, model_id=_DENSE_MODEL_ID)


# ---------------------------------------------------------------------------
# The rules as a derivative
# ---------------------------------------------------------------------------


def test_relp_rules_record_the_three_rules() -> None:
    assert RELP_RULES == {
        "ln_rule": True, "identity_rule": True, "half_rule": True,
        "half_rule_beta": HALF_RULE_BETA, "include_qk_norms": False,
    }
    assert lens_estimator_policy("R")["rules"] == RELP_RULES
    assert lens_estimator_policy("S")["rules"] == RELP_RULES
    assert lens_estimator_policy("J")["rules"] is None
    assert lens_estimator_policy("K")["rules"] is None


@pytest.mark.parametrize("family", ["llama", "gemma3_text"])
def test_transport_mlp_and_norm_maps_follow_the_rules(family: str) -> None:
    """Under ``relp`` the MLP's factors are β·φ(g)·u and β·act(g), and the
    residual norms are diagonal; under ``standard`` they are act'(g)·u,
    act(g) and the full norm Jacobian.  Checked on the oracle's patched
    autograd tangent through one MLP (identity rule + half rule)."""
    model = tiny_dense_model(family)
    spec = model_spec(model)
    block = spec.blocks[1]
    ids = torch.arange(3, 13)
    relp = linearize(spec, ids, rules="relp", skip_first=2)[1]
    standard = linearize(spec, ids, rules="standard", skip_first=2)[1]
    assert relp.attn_in.xhat is None and relp.mlp_in.xhat is None
    assert standard.attn_in.xhat is not None and standard.mlp_in.xhat is not None
    # Head norms keep their true derivative under both rule sets.
    if relp.q_norm is not None:
        assert relp.q_norm.xhat is not None

    mlp = block.mlp
    x = torch.randn(4, spec.d_model, dtype=torch.float64)
    dx = torch.randn(4, spec.d_model, dtype=torch.float64)
    blocks = [b.module for b in spec.blocks]
    with relp_backward_rules(blocks), torch.enable_grad():
        xr = x.clone().requires_grad_(True)
        out = block.module.get_submodule("mlp")(xr)
        (vjp,) = torch.autograd.grad(out, xr, grad_outputs=dx)
    with torch.no_grad():
        g, u = mlp.gate(x), mlp.up(x)
        from drowse.model.blocks import activation, activation_factor

        cg = HALF_RULE_BETA * activation_factor(mlp.act, g) * u
        cu = HALF_RULE_BETA * activation(mlp.act, g)
        gh = dx @ mlp.down.weight
        expected = (cg * gh) @ mlp.gate.weight + (cu * gh) @ mlp.up.weight
    assert torch.allclose(vjp, expected, atol=1e-10)


def test_oracle_patches_keep_forwards_and_restore_modules() -> None:
    model = tiny_dense_model("llama")
    blocks = [b.module for b in model_spec(model).blocks]
    ids = torch.arange(3, 13)[None]
    with torch.no_grad():
        plain = model(ids).logits
        with relp_backward_rules(blocks):
            patched = model(ids).logits
    assert torch.equal(plain, patched)
    assert all("forward" not in module.__dict__ for module in model.modules())


def test_oracle_refuses_a_preexisting_instance_forward() -> None:
    model = tiny_dense_model("llama")
    block = model_spec(model).blocks[0].module
    mlp = block.mlp
    object.__setattr__(mlp, "forward", lambda x: x)
    try:
        with pytest.raises(RelpOracleError, match="instance-level forward"):
            with relp_backward_rules([block]):
                pass
    finally:
        object.__delattr__(mlp, "forward")
    assert all("forward" not in module.__dict__ for module in block.modules())


# ---------------------------------------------------------------------------
# Session-level fits: local/relp lands beside local/default
# ---------------------------------------------------------------------------


def test_relp_fit_lands_beside_standard_and_switches() -> None:
    s = _DenseLensHarness()
    standard = s.lens.fit(_PROMPTS, corpus_spec="test")
    assert standard.n_prompts == len(_PROMPTS)
    relp = s.lens.fit(_PROMPTS, corpus_spec="test", kind="R")
    assert relp.n_prompts == len(_PROMPTS)

    std_sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "j")
    relp_sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "r")
    assert std_sidecar is not None
    assert relp_sidecar is not None
    assert std_sidecar["estimator_policy"] == lens_estimator_policy("J")
    assert relp_sidecar["estimator_policy"] == lens_estimator_policy("R")
    assert relp_sidecar["estimator_policy"]["rules"] == RELP_RULES
    assert std_sidecar["estimator_policy"]["rules"] is None
    assert std_sidecar["kind"] == "J" and relp_sidecar["kind"] == "R"
    assert standard.key == LensKey("J") and relp.key == LensKey("R")

    std_loaded = load_local_lens(_DENSE_MODEL_ID, "j")
    relp_loaded = load_local_lens(_DENSE_MODEL_ID, "r")
    assert std_loaded is not None
    assert relp_loaded is not None
    std_lens, _ = std_loaded
    relp_lens, _ = relp_loaded
    assert relp_lens.key == LensKey("R")
    assert not torch.allclose(std_lens.jacobians[0], relp_lens.jacobians[0])

    sources = {row["source"] for row in list_lens_sources(_DENSE_MODEL_ID)}
    assert {"local:j", "local:r"} <= sources

    # save_lens activates the name it saved: the relp fit ran last.
    assert LENS_SOURCES.active(_DENSE_MODEL_ID, "R") == SourceRef("local", "r")

    use_lens_source(_DENSE_MODEL_ID, "local:j")
    assert LENS_SOURCES.active(_DENSE_MODEL_ID, "R") == SourceRef("local", "j")

    with pytest.raises(ValueError, match="lens kind"):
        s.lens.fit(_PROMPTS, corpus_spec="test", kind="nonsense")
    assert LensKey("R").relp and not LensKey("J").relp


def test_relp_policy_never_resumes_standard_artifact() -> None:
    s = _DenseLensHarness()
    s.lens.fit(_PROMPTS, corpus_spec="test")
    s.lens.fit(_PROMPTS, corpus_spec="test", kind="R")
    std_before = load_local_lens_sidecar(_DENSE_MODEL_ID, "j")
    relp_before = load_local_lens_sidecar(_DENSE_MODEL_ID, "r")
    assert std_before is not None
    assert relp_before is not None

    # Same prompts, standard policy again: an exact no-op against the
    # default artifact, never a resume of (or over) the relp one.
    again = s.lens.fit(_PROMPTS, corpus_spec="test")
    assert again.n_prompts == len(_PROMPTS)

    std_after = load_local_lens_sidecar(_DENSE_MODEL_ID, "j")
    assert std_after is not None
    assert std_after["n_prompts"] == std_before["n_prompts"]
    assert std_after == std_before
    assert load_local_lens_sidecar(_DENSE_MODEL_ID, "r") == relp_before


# ---------------------------------------------------------------------------
# The RelP policy survives checkpoint, resume, promotion and no-op
# ---------------------------------------------------------------------------


_RELP_POLICY = lens_estimator_policy("R")


def _interrupt_relp_fit_after(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a RelP fit that checkpoints every prompt and dies after ``n``."""
    import drowse.lens.fit as jlens_mod

    real_fit = jlens_mod.fit_lens

    def _interrupted(*args: Any, **kwargs: Any) -> Any:
        args = (*args[:2], list(args[2])[:n], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:n]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(jlens_mod, "fit_lens", _interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _DenseLensHarness().lens.fit(
            _PROMPTS, corpus_spec="test", kind="R",
        )
    monkeypatch.setattr(jlens_mod, "fit_lens", real_fit)


def _forbid_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    import drowse.lens.fit as jlens_mod
    import drowse.lens.store as lens_io

    def _fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a finished RelP fit must not refit or rewrite")

    monkeypatch.setattr(jlens_mod, "fit_lens", _fail)
    monkeypatch.setattr(lens_io, "_save_fp32_square_safetensors_atomic", _fail)


def test_relp_checkpoint_records_relp_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _interrupt_relp_fit_after(2, monkeypatch)

    # The pointer the writer produced, read before any loader touches it.
    checkpoint_path = lens_manifest_path(_DENSE_MODEL_ID, "r", checkpoint=True)
    written = json.loads(checkpoint_path.read_text())
    assert written["checkpoint"] is True
    assert written["n_prompts"] == 2
    assert written["kind"] == "R"
    assert written["estimator_policy"] == _RELP_POLICY
    assert load_lens_checkpoint_sidecar(_DENSE_MODEL_ID, "r") == written


def test_interrupted_relp_fit_resumes_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_mod

    full = _DenseLensHarness().lens.fit(
        _PROMPTS, corpus_spec="test", kind="R",
    )
    remove_lens(_DENSE_MODEL_ID, "r")
    _interrupt_relp_fit_after(2, monkeypatch)

    real_fit = jlens_mod.fit_lens
    consumed: list[int] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        consumed.append(len(list(args[2])))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_lens", _counting)
    messages: list[str] = []
    resumed = _DenseLensHarness().lens.fit(
        _PROMPTS, corpus_spec="test", kind="R",
        on_progress=messages.append,
    )
    assert any("checkpoint at 2 prompts" in m for m in messages)
    assert consumed == [len(_PROMPTS) - 2]
    assert resumed.n_prompts == len(_PROMPTS)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_promoted_relp_final_carries_relp_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    promoted: list[bool] = []
    real_promote = lens_io.promote_lens_checkpoint

    def _spy(*args: Any, **kwargs: Any) -> bool:
        promoted.append(real_promote(*args, **kwargs))
        return promoted[-1]

    monkeypatch.setattr(lens_io, "promote_lens_checkpoint", _spy)
    _DenseLensHarness().lens.fit(
        _PROMPTS, corpus_spec="test", kind="R",
        checkpoint_every=len(_PROMPTS),
    )
    assert promoted == [True]
    final = json.loads(lens_manifest_path(_DENSE_MODEL_ID, "r").read_text())
    assert final["checkpoint"] is False
    assert final["kind"] == "R"
    assert final["estimator_policy"] == _RELP_POLICY
    assert load_lens_checkpoint_sidecar(_DENSE_MODEL_ID, "r") is None


def test_finished_relp_fit_rerun_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _DenseLensHarness().lens.fit(
        _PROMPTS, corpus_spec="test", kind="R",
        checkpoint_every=len(_PROMPTS),
    )
    before = load_local_lens_sidecar(_DENSE_MODEL_ID, "r")
    assert before is not None

    _forbid_estimator(monkeypatch)
    messages: list[str] = []
    again = _DenseLensHarness().lens.fit(
        _PROMPTS, corpus_spec="test", kind="R",
        on_progress=messages.append,
    )
    assert any("nothing to do" in m for m in messages)
    assert again.n_prompts == len(_PROMPTS)
    assert load_local_lens_sidecar(_DENSE_MODEL_ID, "r") == before


def test_cli_preflight_reports_finished_relp_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.corpus import HfLensCorpus

    corpus = HfLensCorpus("repo", "config", "a" * 40)
    spec = corpus.spec
    session = _DenseLensHarness()
    setattr(session.model, "_drowse_source_fingerprint", "source-a")
    session.lens.fit(
        _PROMPTS, corpus_spec=spec, kind="R",
        checkpoint_every=len(_PROMPTS),
    )
    monkeypatch.setattr(
        "drowse.model.fingerprint.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    args = argparse.Namespace(
        force=False, model=_DENSE_MODEL_ID, quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=len(_PROMPTS), kind="r",
        sites=("out", "mid"),
    )
    assert lens_fit_is_noop(args, None, hf_corpus=corpus)
    args.kind = "j"  # the standard fit has its own, absent, artifact
    assert not lens_fit_is_noop(args, None, hf_corpus=corpus)


def test_lens_writers_bind_the_r_name_to_the_r_kind() -> None:
    """``local:r`` holds an R-lens and nothing else, and vice versa."""
    lens = JacobianLens({0: torch.eye(DENSE_D)}, n_prompts=1, d_model=DENSE_D)
    prov = LensProvenance(
        corpus_spec="test", corpus_sha256="0" * 64, seq_len=128, dim_batch=8,
    )
    with pytest.raises(ValueError, match="local:r"):
        save_lens(lens, _DENSE_MODEL_ID, prov, name="r")
    relp = JacobianLens(
        {0: torch.eye(DENSE_D)}, n_prompts=1, d_model=DENSE_D, key=LensKey("R"),
    )
    with pytest.raises(ValueError, match="local:j"):
        save_lens(relp, _DENSE_MODEL_ID, prov, name="j")
    with pytest.raises(ValueError, match="local:r"):
        save_lens_checkpoint(
            {"out": {0: torch.eye(DENSE_D)}}, 1, DENSE_D, _DENSE_MODEL_ID, prov,
            kind="J", name="r", consumed_prefix_sha256="1" * 64,
        )


def test_fit_applies_the_relp_rules_for_an_r_kind() -> None:
    """``fit_lens(kind="R")`` takes the RelP derivative, ``"J"`` the true one:
    the two differ, each matches the oracle under its own rules, and the
    model's modules are untouched after."""
    from drowse.lens.fit import fit_lens
    from tests.support.lens_oracle import oracle_lens

    session = _DenseLensHarness()
    model = session.model
    tokenizer = session.tokenizer
    r_lens = fit_lens(model, tokenizer, _PROMPTS[:2], kind="R", sites=["out"]).lens
    j_lens = fit_lens(model, tokenizer, _PROMPTS[:2], kind="J", sites=["out"]).lens
    assert r_lens.key == LensKey("R") and j_lens.key == LensKey("J")
    rows = [tokenizer.encode(p) for p in _PROMPTS[:2]]
    layers = {"out": r_lens.source_layers}
    for kind, lens in (("R", r_lens), ("J", j_lens)):
        ref = oracle_lens(model, rows, kind=kind, layers=layers, skip_first=16)
        for layer in lens.source_layers:
            assert torch.allclose(lens.jacobians[layer], ref["out"][layer], atol=1e-5)
    for layer in r_lens.source_layers:
        assert not torch.allclose(r_lens.jacobians[layer], j_lens.jacobians[layer])
    assert all("forward" not in mod.__dict__ for mod in model.modules())
