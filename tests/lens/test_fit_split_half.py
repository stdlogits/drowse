"""Split-half records and per-site fits through the store and the lifecycle.

The manifest carries ``sites.<site>.split_half`` (``{layer: value}``, every
layer of the site) when a fit measured it; checkpoints also carry
``sites.<site>.odd``, the odd-half mean shards a resumed fit continues from,
which promotion drops.  Through the lifecycle: an interrupted fit resumes
with the same split-half as an uninterrupted one, a top-up keeps the old
site's values and measures the new one, a durable corpus extension (no
halves) records none, neither an input-kind fit nor a ``mid``-only output
fit touches the output selection or resident lens, and the model is
verified once per fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

import drowse.lens.fit as fit_module
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.store import (
    LensProvenance,
    lens_manifest_path,
    lens_split_half,
    load_lens_checkpoint,
    load_lens_checkpoint_odd,
    load_lens_checkpoint_sidecar,
    load_local_lens,
    load_local_lens_sidecar,
    promote_lens_checkpoint,
    remove_lens,
    save_lens,
    save_lens_checkpoint,
)
from tests.support.lens import LENS_PROMPTS, TOY_MODEL_ID, LensHarness

pytestmark = pytest.mark.usefixtures("isolated_home")

_D = 4
_MODEL = "toy/split-half"
_DOCS = [*LENS_PROMPTS, "lmkjihgfedcbamlkjihgfedcbalm"]
_OUT = ("out",)


def _prov(**kw: Any) -> LensProvenance:
    return LensProvenance(
        corpus_spec="test", corpus_sha256="a" * 64, seq_len=64, dim_batch=8,
        model_fingerprint="fp", **kw,
    )


def _lens(kind: str = "R", n: int = 3) -> JacobianLens:
    gen = torch.Generator().manual_seed(0)
    return JacobianLens(
        {0: torch.randn(_D, _D, generator=gen), 1: torch.randn(_D, _D, generator=gen)},
        mid={0: torch.randn(_D, _D, generator=gen)},
        n_prompts=n, d_model=_D, key=LensKey(kind),
    )


# -- the manifest ----------------------------------------------------------------------

def test_split_half_is_recorded_per_site_only_when_complete() -> None:
    save_lens(
        _lens(), _MODEL, _prov(),
        split_half={"out": {0: 0.25, 1: 0.5}, "mid": {}},
    )
    sidecar = load_local_lens_sidecar(_MODEL, "r")
    assert sidecar is not None
    assert sidecar["sites"]["out"]["split_half"] == {"0": 0.25, "1": 0.5}
    assert "split_half" not in sidecar["sites"]["mid"]
    assert lens_split_half(sidecar, "out") == {0: 0.25, 1: 0.5}
    assert lens_split_half(sidecar, "mid") == {}
    # An imported or older manifest without the field reads as unmeasured.
    save_lens(_lens(), _MODEL, _prov())
    sidecar = load_local_lens_sidecar(_MODEL, "r")
    assert sidecar is not None and lens_split_half(sidecar, "out") == {}


@pytest.mark.parametrize("bad", [
    {"0": 0.1},                      # misses layer 1
    {"0": 0.1, "1": 0.2, "2": 0.3},  # names a layer the site lacks
    {"0": -0.1, "1": 0.2},
    {"0": True, "1": 0.2},
    {"0": "0.1", "1": 0.2},
    {"0": float("nan"), "1": 0.2},
])
def test_an_invalid_split_half_reads_as_no_lens(bad: dict[str, Any]) -> None:
    save_lens(_lens(), _MODEL, _prov(), split_half={"out": {0: 0.1, 1: 0.2}})
    path = lens_manifest_path(_MODEL, "r")
    manifest = json.loads(path.read_text())
    manifest["sites"]["out"]["split_half"] = bad
    path.write_text(json.dumps(manifest))
    assert load_local_lens_sidecar(_MODEL, "r") is None


def test_odd_half_shards_are_checkpoint_only() -> None:
    save_lens(_lens(), _MODEL, _prov())
    path = lens_manifest_path(_MODEL, "r")
    manifest = json.loads(path.read_text())
    manifest["sites"]["out"]["odd"] = {"tensor_files": {}, "tensor_sha256": {}}
    path.write_text(json.dumps(manifest))
    assert load_local_lens_sidecar(_MODEL, "r") is None


def _checkpoint(n: int, *, odd: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = torch.Generator().manual_seed(n)
    total = {
        "out": {0: torch.randn(_D, _D, generator=gen), 1: torch.randn(_D, _D, generator=gen)},
        "mid": {0: torch.randn(_D, _D, generator=gen)},
    }
    odd_sums = {
        site: {layer: torch.randn(_D, _D, generator=gen) for layer in m}
        for site, m in total.items()
    }
    save_lens_checkpoint(
        total, n, _D, _MODEL, _prov(), kind="R", consumed_prefix_sha256="b" * 64,
        odd_sums=odd_sums if odd else None,
        split_half={"out": {0: 0.1, 1: 0.2}, "mid": {0: 0.3}},
    )
    return total, odd_sums


def test_checkpoint_odd_halves_round_trip_and_promotion_drops_them() -> None:
    _total, odd_sums = _checkpoint(5)
    sidecar = load_lens_checkpoint_sidecar(_MODEL, "r")
    assert sidecar is not None and set(sidecar["sites"]["out"]["odd"]) == {"tensor_files", "tensor_sha256"}
    odd = load_lens_checkpoint_odd(_MODEL, "r", sidecar)
    assert odd is not None
    for site, matrices in odd_sums.items():
        for layer, matrix in matrices.items():
            assert torch.allclose(odd[site][layer], matrix / 2)  # n_odd = 5 // 2
    directory = lens_manifest_path(_MODEL, "r").parent
    odd_files = {
        name for entry in sidecar["sites"].values() for name in entry["odd"]["tensor_files"].values()
    }
    assert all((directory / name).exists() for name in odd_files)
    assert promote_lens_checkpoint(
        _MODEL, name="r", n_prompts=5, site_layers={"out": [0, 1], "mid": [0]},
        corpus_sha256="a" * 64, corpus_hash_kind="token_ids_v1", seq_len=64,
        d_model=_D, model_fingerprint="fp",
    )
    final = load_local_lens_sidecar(_MODEL, "r")
    assert final is not None
    assert all("odd" not in entry for entry in final["sites"].values())
    assert lens_split_half(final, "out") == {0: 0.1, 1: 0.2}
    assert not any((directory / name).exists() for name in odd_files)


def test_odd_halves_follow_their_checkpoint() -> None:
    _checkpoint(5)
    first = load_lens_checkpoint_sidecar(_MODEL, "r")
    assert first is not None
    _checkpoint(6)  # replaced: the earlier sidecar no longer names live shards
    assert load_lens_checkpoint_odd(_MODEL, "r", first) is None
    current = load_lens_checkpoint_sidecar(_MODEL, "r")
    assert current is not None and load_lens_checkpoint_odd(_MODEL, "r", current) is not None
    _checkpoint(7, odd=False)
    bare = load_lens_checkpoint_sidecar(_MODEL, "r")
    assert bare is not None and load_lens_checkpoint_odd(_MODEL, "r", bare) is None


def test_a_one_document_checkpoint_writes_no_odd_half() -> None:
    _checkpoint(1)
    sidecar = load_lens_checkpoint_sidecar(_MODEL, "r")
    assert sidecar is not None and "odd" not in sidecar["sites"]["out"]


# -- through the lifecycle ---------------------------------------------------------------

def _interrupt_after(n: int, monkeypatch: pytest.MonkeyPatch, **fit: Any) -> None:
    real_fit = fit_module.fit_lens

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        args = (*args[:2], list(args[2])[:n], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:n]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(fit_module, "fit_lens", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        LensHarness().lens.fit(_DOCS, kind="R", **fit)
    monkeypatch.setattr(fit_module, "fit_lens", real_fit)


def test_a_resumed_fit_keeps_its_split_half(monkeypatch: pytest.MonkeyPatch) -> None:
    LensHarness().lens.fit(_DOCS, kind="R", force=True)
    whole = load_local_lens_sidecar(TOY_MODEL_ID, "r")
    assert whole is not None
    remove_lens(TOY_MODEL_ID, "r")
    _interrupt_after(3, monkeypatch)
    checkpoint = load_lens_checkpoint(TOY_MODEL_ID, "r")
    assert checkpoint is not None and checkpoint[0].n_prompts == 3
    messages: list[str] = []
    LensHarness().lens.fit(_DOCS, kind="R", on_progress=messages.append)
    assert any("checkpoint at 3 prompts" in m for m in messages)
    resumed = load_local_lens_sidecar(TOY_MODEL_ID, "r")
    assert resumed is not None
    for site in ("out", "mid"):
        want, got = lens_split_half(whole, site), lens_split_half(resumed, site)
        assert set(want) == set(got) and want
        for layer in want:
            assert math.isclose(got[layer], want[layer], rel_tol=1e-4)


def test_a_durable_corpus_extension_records_no_split_half() -> None:
    LensHarness().lens.fit(_DOCS[:2], kind="J", sites=_OUT, force=True)
    first = load_local_lens_sidecar(TOY_MODEL_ID, "j")
    assert first is not None and lens_split_half(first, "out")
    messages: list[str] = []
    grown = LensHarness().lens.fit(_DOCS, kind="J", sites=_OUT, on_progress=messages.append)
    assert grown.n_prompts == len(_DOCS)
    assert any("resuming from 2 prompts" in m for m in messages)
    sidecar = load_local_lens_sidecar(TOY_MODEL_ID, "j")
    assert sidecar is not None and lens_split_half(sidecar, "out") == {}


def test_a_site_topup_keeps_the_old_site_and_measures_the_new() -> None:
    LensHarness().lens.fit(_DOCS, kind="R", sites=_OUT, force=True)
    before = load_local_lens_sidecar(TOY_MODEL_ID, "r")
    assert before is not None and set(before["sites"]) == {"out"}
    messages: list[str] = []
    both = LensHarness().lens.fit(_DOCS, kind="R", on_progress=messages.append)
    assert any("fitting missing R layers mid [0, 1, 2]" in m for m in messages)
    assert both.sites == ("out", "mid")
    after = load_local_lens_sidecar(TOY_MODEL_ID, "r")
    assert after is not None
    assert after["sites"]["out"]["tensor_files"] == before["sites"]["out"]["tensor_files"]
    assert lens_split_half(after, "out") == lens_split_half(before, "out")
    assert set(lens_split_half(after, "mid")) == {0, 1, 2}
    whole = LensHarness().lens.fit(_DOCS, kind="R", force=True)
    for layer in (0, 1, 2):
        assert torch.allclose(both.matrix(layer, "mid"), whole.matrix(layer, "mid"), atol=1e-6)


def test_an_input_kind_fit_leaves_the_output_resident_alone() -> None:
    harness = LensHarness()
    resident = harness.lens.fit(_DOCS[:2], kind="R", force=True)
    assert harness.lens.resident_lens is resident
    s_lens = harness.lens.fit(_DOCS[:2], kind="S", sites=["mid"], force=True)
    assert s_lens.key.kind == "S" and s_lens.sites == ("mid",)
    assert harness.lens.resident_lens is resident
    loaded = load_local_lens(TOY_MODEL_ID, "s")
    assert loaded is not None and loaded[0].sites == ("mid",)
    # A second identical input fit is a no-op.
    messages: list[str] = []
    harness.lens.fit(_DOCS[:2], kind="S", sites=["mid"], on_progress=messages.append)
    assert any("nothing to do" in m for m in messages)


def test_a_mid_only_output_fit_leaves_the_output_source_and_readout_alone() -> None:
    """The resident readout reads ``out``: an R fit at ``mid`` only publishes
    ``local:r`` unselected and leaves the selected output source, the
    resident lens and live readout as they were."""
    from drowse.base.source_registry import LOCAL, SourceRef
    from drowse.lens.sources import LENS_SOURCES

    harness = LensHarness()
    resident = harness.lens.fit(_DOCS[:2], kind="J", force=True)
    harness.lens.set_live(True)
    live = harness.lens.live_config()
    assert harness.lens.resident_lens is resident and live is not None
    mid_only = harness.lens.fit(_DOCS[:2], kind="R", sites=["mid"], force=True)
    assert mid_only.sites == ("mid",)
    assert LENS_SOURCES.active(TOY_MODEL_ID, "output") == SourceRef(LOCAL, "j")
    assert harness.lens.resident_lens is resident
    assert harness.lens.live_config() == live
    assert harness.lens.readout(_DOCS[0], top_k=1)
    sidecar = load_local_lens_sidecar(TOY_MODEL_ID, "r")
    assert sidecar is not None and list(sidecar["sites"]) == ["mid"]


def test_a_lifecycle_fit_verifies_the_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import drowse.model.blocks as blocks

    real = blocks.verify_model_spec
    calls: list[Any] = []

    def counting(model: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(model)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(blocks, "verify_model_spec", counting)
    LensHarness().lens.fit(_DOCS, kind="S", force=True)
    assert len(calls) == 1


def test_the_manifest_split_half_matches_the_fit() -> None:
    harness = LensHarness()
    harness.lens.fit(_DOCS, kind="K", sites=["out"], force=True)
    sidecar = load_local_lens_sidecar(TOY_MODEL_ID, "k")
    assert sidecar is not None
    direct = fit_module.fit_lens(
        harness.model, harness.tokenizer, _DOCS, kind="K", sites=["out"],
    )
    recorded = lens_split_half(sidecar, "out")
    assert set(recorded) == set(direct.split_half["out"])
    for layer, value in direct.split_half["out"].items():
        assert math.isclose(recorded[layer], value, rel_tol=1e-6)
    raw = json.loads(Path(lens_manifest_path(TOY_MODEL_ID, "k")).read_text())
    assert raw["format_version"] == 7
