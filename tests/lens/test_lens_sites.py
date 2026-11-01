"""One lens object across kinds (J/R/K/S) and sites (out/mid), and its v7 store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file

from drowse.base.source_registry import LOCAL, SourceRef
from drowse.lens.jlens import JacobianLens, JacobianLensError, LensNotFittedError
from drowse.lens.key import LensKey
from drowse.lens.sources import LENS_SOURCES
from drowse.lens.store import (
    LensProvenance,
    lens_layers,
    lens_manifest_path,
    load_lens,
    load_lens_checkpoint,
    load_lens_sidecar,
    load_local_lens,
    promote_lens_checkpoint,
    remove_subsumed_lens_checkpoint,
    save_lens,
    save_lens_at,
    save_lens_checkpoint,
)

_MODEL = "test-org/tiny-model"
_D = 6


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_PROV = LensProvenance(
    corpus_spec="test-corpus", corpus_sha256=_digest("abc"), seq_len=64, dim_batch=8,
)


def _mats(layers: range | list[int], seed: int, d_in: int = _D) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {layer: torch.randn(_D, d_in, generator=gen) for layer in layers}


def _two_site(kind: str, n_prompts: int = 5) -> JacobianLens:
    """An R-shaped layout: ``out`` below the last layer, ``mid`` at every layer."""
    return JacobianLens(
        _mats(range(3), 1), mid=_mats(range(4), 2),
        n_prompts=n_prompts, d_model=_D, key=LensKey(kind),
    )


# -- the object --------------------------------------------------------------------

def test_a_lens_holds_its_kind_and_sites() -> None:
    lens = _two_site("R")
    assert (lens.kind, lens.sites) == ("R", ("out", "mid"))
    assert lens.site_key("mid") == LensKey("R", "mid")
    assert lens.layers("out") == [0, 1, 2] and lens.layers("mid") == [0, 1, 2, 3]
    assert lens.source_layers == [0, 1, 2]
    assert lens.jacobians is lens.matrices("out")
    assert all(m.dtype == torch.float32 for m in lens.matrices("mid").values())
    assert "out=[0..2] (3 layers)" in repr(lens) and "mid=[0..3] (4 layers)" in repr(lens)

    mid_only = JacobianLens(mid=_mats(range(2), 3), n_prompts=1, d_model=_D, key=LensKey("S"))
    assert mid_only.sites == ("mid",)


def test_a_missing_site_is_a_404_that_never_falls_back() -> None:
    mid_only = JacobianLens(mid=_mats(range(2), 3), n_prompts=1, d_model=_D, key=LensKey("R"))
    with pytest.raises(LensNotFittedError, match="no 'out' site") as info:
        mid_only.transport(torch.zeros(_D), 0)
    assert info.value.user_message()[0] == 404
    with pytest.raises(LensNotFittedError):
        _ = mid_only.source_layers
    with pytest.raises(LensNotFittedError, match="layer 5 not in fitted R:mid"):
        mid_only.transport(torch.zeros(_D), 5, site="mid")


def test_output_kinds_transport_and_input_kinds_pull_back() -> None:
    J = _two_site("J")
    h = torch.randn(2, _D)
    torch.testing.assert_close(J.transport(h, 1, site="mid"), h @ J.matrices("mid")[1].T)

    K_out = _mats(range(2), 7, d_in=4)
    S = JacobianLens(K_out, n_prompts=1, d_model=_D, d_emb=4, key=LensKey("S"))
    assert S.matrix_shape == (_D, 4)
    c = torch.randn(3, _D)
    e = torch.randn(3, 4)
    torch.testing.assert_close(S.pullback(c, 1), c @ K_out[1])          # [3, d_emb]
    torch.testing.assert_close(S.footprint(e, 1), e @ K_out[1].T)       # [3, d]
    # The pullback and footprint are adjoint: c·(K ẽ) == (c K)·ẽ.
    torch.testing.assert_close(
        (c * S.footprint(e, 0)).sum(-1), (S.pullback(c, 0) * e).sum(-1),
    )
    with pytest.raises(JacobianLensError, match="transport reads an R-family lens"):
        S.transport(c, 0)
    with pytest.raises(JacobianLensError, match="pullback reads an S-family lens"):
        J.pullback(c, 0)
    with pytest.raises(JacobianLensError):
        S.token_direction(0, torch.randn(10, _D))


@pytest.mark.parametrize("kind,shape,d_emb", [
    ("J", (_D, _D + 1), None),
    ("R", (_D + 1, _D + 1), None),
    ("S", (_D, _D), 4),
    ("K", (_D, 4), None),
])
def test_matrix_shapes_are_checked_per_kind(kind: str, shape: tuple[int, int], d_emb: int | None) -> None:
    with pytest.raises(ValueError, match="matrices"):
        JacobianLens({0: torch.zeros(shape)}, n_prompts=1, d_model=_D, d_emb=d_emb, key=LensKey(kind))


def test_construction_refusals() -> None:
    with pytest.raises(ValueError, match="pass R:mid matrices as mid="):
        JacobianLens(_mats(range(1), 0), n_prompts=1, d_model=_D, key=LensKey("R", "mid"))
    with pytest.raises(ValueError, match="no separate d_emb"):
        JacobianLens(_mats(range(1), 0), n_prompts=1, d_model=_D, d_emb=4, key=LensKey("J"))


def test_select_and_union_work_site_by_site() -> None:
    lens = _two_site("R")
    both = lens.select_layers({"out": [1, 2], "mid": [1, 2]})
    assert both.sites == ("out", "mid") and both.layers("mid") == [1, 2]
    assert both.matrices("mid")[1] is lens.matrices("mid")[1]
    # Each site is asked for its own layers: the last layer lives at mid only.
    asymmetric = lens.select_layers({"out": [0, 1, 2], "mid": [0, 1, 2, 3]})
    assert asymmetric.layers("out") == [0, 1, 2] and asymmetric.layers("mid") == [0, 1, 2, 3]
    with pytest.raises(LensNotFittedError, match="R:out"):
        lens.select_layers({"out": [3]})  # the mid site has it, the out site does not
    mid = lens.select_layers({"mid": [3]})
    assert mid.sites == ("mid",) and mid.layers("mid") == [3]
    with pytest.raises(LensNotFittedError, match="no 'mid' site"):
        JacobianLens(_mats(range(3), 1), n_prompts=5, d_model=_D, key=LensKey("R")).select_layers({"mid": [0]})

    low = JacobianLens(_mats([0], 4), mid=_mats([0], 5), n_prompts=5, d_model=_D, key=LensKey("R"))
    high = JacobianLens(_mats([1, 2], 6), n_prompts=5, d_model=_D, key=LensKey("R"))
    union = JacobianLens.union_layers([low, high])
    assert union.layers("out") == [0, 1, 2] and union.layers("mid") == [0]
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.union_layers([low, JacobianLens(_mats([1], 6), n_prompts=5, d_model=_D, key=LensKey("J"))])


# -- the store ---------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["R", "J", "S", "K"])
def test_two_site_lenses_round_trip_under_kind_carrying_names(kind: str) -> None:
    lens = _two_site(kind)
    path = save_lens(lens, _MODEL, _PROV)
    assert path.parent.name == kind.lower()
    sidecar = json.loads(path.read_text())
    assert sidecar["kind"] == kind and set(sidecar["sites"]) == {"out", "mid"}
    assert lens_layers(sidecar, "mid") == [0, 1, 2, 3]
    names = sorted(p.name.split(".gen-")[0] for p in path.parent.glob("*.safetensors"))
    assert names == [*(f"mid.layer-{l}" for l in range(4)), *(f"out.layer-{l}" for l in range(3))]

    loaded = load_local_lens(_MODEL, kind.lower())
    assert loaded is not None
    back, _ = loaded
    assert (back.kind, back.sites) == (kind, ("out", "mid"))
    for site in ("out", "mid"):
        for layer, matrix in lens.matrices(site).items():
            torch.testing.assert_close(back.matrices(site)[layer], matrix)


def test_input_kinds_publish_into_the_input_selection() -> None:
    save_lens(_two_site("R"), _MODEL, _PROV)
    save_lens(_two_site("S"), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "R") == SourceRef(LOCAL, "r")
    assert LENS_SOURCES.active(_MODEL, "S") == SourceRef(LOCAL, "s")
    output, inp = load_lens(_MODEL), load_lens(_MODEL, "S")
    assert output is not None and inp is not None
    assert (output[0].kind, inp[0].kind) == ("R", "S")
    sidecar = load_lens_sidecar(_MODEL, "S")
    assert sidecar is not None and sidecar["kind"] == "S"


def test_save_lens_at_writes_every_site_without_selecting(tmp_path: Path) -> None:
    path = save_lens_at(_two_site("S"), tmp_path / "pack", _PROV)
    assert set(json.loads(path.read_text())["sites"]) == {"out", "mid"}
    assert LENS_SOURCES.active(_MODEL, "S") is None


def test_save_lens_at_keeps_an_out_only_pack_in_the_browser_readers_shape(tmp_path: Path) -> None:
    """drowse.ai's v7 reader takes exactly one ``sites.out`` holding exactly
    ``layers``/``tensor_files``/``tensor_sha256``.  The hosted packaging path
    (an output lens at ``out`` only, no split-half) must keep writing that."""
    lens = JacobianLens(_mats(range(3), 1), n_prompts=5, d_model=_D, key=LensKey("R"))
    manifest = json.loads(save_lens_at(lens, tmp_path / "pack", _PROV).read_text())
    assert manifest["format_version"] == 7
    assert list(manifest["sites"]) == ["out"]
    assert sorted(manifest["sites"]["out"]) == ["layers", "tensor_files", "tensor_sha256"]


def _mid_only(kind: str) -> JacobianLens:
    return JacobianLens(mid=_mats(range(4), 2), n_prompts=5, d_model=_D, key=LensKey(kind))


def test_publishing_a_lens_without_out_leaves_the_selection_alone() -> None:
    """Both families' probes, gates and atoms read ``out``: a lens of either
    family fitted at ``mid`` only publishes, but never displaces (or
    becomes) its family's selected source."""
    save_lens(JacobianLens(_mats(range(3), 1), n_prompts=5, d_model=_D, key=LensKey("J")), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "R") == SourceRef(LOCAL, "j")
    save_lens(_mid_only("R"), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "R") == SourceRef(LOCAL, "j")
    loaded = load_local_lens(_MODEL, "r")
    assert loaded is not None and loaded[0].sites == ("mid",)
    # The promotion path follows the same rule.
    _checkpoint("R", _mid_only("R"))
    assert promote_lens_checkpoint(
        _MODEL, name="r", n_prompts=5, site_layers={"mid": [0, 1, 2, 3]},
        corpus_sha256=_PROV.corpus_sha256, corpus_hash_kind="token_ids_v1",
        seq_len=64, d_model=_D, model_fingerprint=cast(str, None),
    )
    assert LENS_SOURCES.active(_MODEL, "R") == SourceRef(LOCAL, "j")
    # The S family follows the same rule: a mid-only S lens selects nothing,
    # and never displaces an S-family source that holds out.
    save_lens(_mid_only("S"), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "S") is None
    save_lens(JacobianLens(_mats(range(3), 1), n_prompts=5, d_model=_D, key=LensKey("K")), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "S") == SourceRef(LOCAL, "k")
    save_lens(_mid_only("S"), _MODEL, _PROV)
    assert LENS_SOURCES.active(_MODEL, "S") == SourceRef(LOCAL, "k")
    assert LENS_SOURCES.active(_MODEL, "R") == SourceRef(LOCAL, "j")


def test_a_d_emb_v7_cannot_record_is_refused() -> None:
    lens = JacobianLens(_mats(range(1), 0, d_in=4), n_prompts=1, d_model=_D, d_emb=4, key=LensKey("S"))
    with pytest.raises(ValueError, match="d_emb == d_model"):
        save_lens(lens, _MODEL, _PROV)


def test_a_wrong_shape_mid_shard_loads_as_no_lens() -> None:
    path = save_lens(_two_site("R"), _MODEL, _PROV)
    sidecar = json.loads(path.read_text())
    shard = path.parent / sidecar["sites"]["mid"]["tensor_files"]["2"]
    save_file({"layer_2": torch.zeros(_D, _D + 1)}, str(shard))
    assert load_local_lens(_MODEL, "r") is None


def test_selective_load_asks_each_site_for_its_own_layers() -> None:
    """An R lens's out site stops below the last layer and its mid site covers it.

    A request naming the last layer at mid only is a valid load, not an
    unusable artifact; a request the artifact cannot meet is a typed 404,
    not "re-fit".
    """
    from drowse.lens.store import _load_lens_verified

    save_lens(_two_site("R"), _MODEL, _PROV)  # out [0..2], mid [0..3]
    loaded = _load_lens_verified(_MODEL, name="r", requested_layers={"out": {1, 2}, "mid": {1, 2}})
    assert loaded is not None
    lens, _sidecar, proof = loaded
    assert lens.layers("out") == [1, 2] and lens.layers("mid") == [1, 2]
    assert proof is None  # a partial load proves nothing about the rest

    asymmetric = _load_lens_verified(
        _MODEL, name="r", requested_layers={"out": {0, 1, 2}, "mid": {0, 1, 2, 3}},
    )
    assert asymmetric is not None
    lens, _sidecar, proof = asymmetric
    assert lens.layers("out") == [0, 1, 2] and lens.layers("mid") == [0, 1, 2, 3]
    assert proof is not None  # the request names the whole artifact

    last = _load_lens_verified(_MODEL, name="r", requested_layers={"mid": {3}})
    assert last is not None and last[0].sites == ("mid",) and last[0].layers("mid") == [3]

    with pytest.raises(LensNotFittedError, match=r"layers \[3\] not in fitted R:out") as info:
        _load_lens_verified(_MODEL, name="r", requested_layers={"out": {3}})
    assert info.value.user_message()[0] == 404
    save_lens(JacobianLens(_mats(range(3), 1), n_prompts=5, d_model=_D, key=LensKey("R")), _MODEL, _PROV)
    with pytest.raises(LensNotFittedError, match="no 'mid' site"):
        _load_lens_verified(_MODEL, name="r", requested_layers={"mid": {0}})


def _checkpoint(kind: str, lens: JacobianLens, **extra: Any) -> Path:
    sums = {
        site: {layer: m * lens.n_prompts for layer, m in lens.matrices(site).items()}
        for site in lens.sites
    }
    return save_lens_checkpoint(
        sums, lens.n_prompts, _D, _MODEL, _PROV, kind=kind,
        consumed_prefix_sha256=_digest("prefix"), **extra,
    )


def test_a_two_site_checkpoint_round_trips_and_promotes() -> None:
    partial = _two_site("S", n_prompts=4)
    _checkpoint("S", partial)
    loaded = load_lens_checkpoint(_MODEL, "s")
    assert loaded is not None and loaded[0].sites == ("out", "mid")
    torch.testing.assert_close(loaded[0].matrices("mid")[3], partial.matrices("mid")[3])

    def promote(site_layers: dict[str, list[int]]) -> bool:
        return promote_lens_checkpoint(
            _MODEL, name="s", n_prompts=4, site_layers=site_layers,
            corpus_sha256=_PROV.corpus_sha256, corpus_hash_kind="token_ids_v1",
            seq_len=64, d_model=_D, model_fingerprint=cast(str, None),
        )

    assert not promote({"out": [0, 1, 2]})
    assert promote({"out": [0, 1, 2], "mid": [0, 1, 2, 3]})
    assert not lens_manifest_path(_MODEL, "s", checkpoint=True).exists()
    final = load_local_lens(_MODEL, "s")
    assert final is not None and final[0].layers("mid") == [0, 1, 2, 3]
    assert LENS_SOURCES.active(_MODEL, "S") == SourceRef(LOCAL, "s")


def test_a_checkpoint_is_subsumed_only_when_every_site_is() -> None:
    final = _two_site("R", n_prompts=6)
    save_lens(final, _MODEL, _PROV)
    # A checkpoint whose mid site reaches a layer the final lacks is not subsumed.
    wider = JacobianLens(_mats(range(1), 8), mid=_mats(range(5), 9), n_prompts=4, d_model=_D, key=LensKey("R"))
    _checkpoint("R", wider)
    assert not remove_subsumed_lens_checkpoint(_MODEL, name="r")
    narrower = JacobianLens(_mats(range(1), 8), mid=_mats(range(2), 9), n_prompts=4, d_model=_D, key=LensKey("R"))
    _checkpoint("R", narrower)
    assert remove_subsumed_lens_checkpoint(_MODEL, name="r")
