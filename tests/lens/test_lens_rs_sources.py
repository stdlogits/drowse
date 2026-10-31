"""The rs-lenses providers (``rs-r``, ``rs-s``): multi-site bindings over a
synthetic Hub repository laid out like ``stdlogits/rs-lenses``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.base.atomic import write_json_atomic
from drowse.base.source_registry import SourceRef
from drowse.lens.jlens import LensNotFittedError
from drowse.lens.key import LensKey

_D = 4
_LAYERS = 3


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "drowse"))


def _provenance(kind: str, site: str, *, model_id: str = "org/model") -> dict[str, Any]:
    config = {
        "estimator": "relp",
        "rules": {"ln_rule": "rsqrt held", "half_rule": 0.5},
        "lens": kind,
        "target": site,
    }
    return {
        "model_id": model_id,
        "dataset_id": "HuggingFaceFW/dclm_100BT-shuffled@abc",
        "rows": [0, 7],
        "rows_sha256": "f" * 64,
        "n_prompts": 7,
        "t_max": 128,
        "skip_first": 16,
        "config_json": json.dumps(config),
        "split_half": [0.01] * _LAYERS,
        "date": "2026-10-29",
        **({"target_layer": _LAYERS - 1} if kind == "R" else {}),
    }


def _payload(
    kind: str, site: str, *, scale: float = 1.0, provenance: bool = True, **extra: Any,
) -> dict[str, Any]:
    """One rs-lenses file: R files hold ``J``/``source_layers`` (``out``
    includes the identity at the target layer), S files ``K``/``layers``."""
    matrices = {L: scale * (L + 1) * torch.eye(_D) for L in range(_LAYERS)}
    if kind == "R":
        body: dict[str, Any] = {"J": matrices, "source_layers": list(range(_LAYERS))}
    else:
        body = {"K": matrices, "layers": list(range(_LAYERS))}
    body.update({"n_prompts": 7, "d_model": _D})
    if provenance:
        body["provenance"] = _provenance(kind, site)
    body.update(extra)
    return body


def _hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Serve ``files`` (repo path -> payload) as the pinned repository."""
    from drowse.lens import sources

    root = tmp_path / "hf-cache"
    root.mkdir(exist_ok=True)
    local: dict[str, str] = {}
    for index, (repo_path, payload) in enumerate(files.items()):
        path = root / f"{index}.pt"
        torch.save(payload, path)
        local[repo_path] = str(path)
    api = SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(sha="rs-commit"),
        list_repo_files=lambda repo, revision: sorted(local) + ["README.md"],
    )
    monkeypatch.setattr(sources, "_hf_api", lambda: api)
    monkeypatch.setattr(
        sources, "_hf_hub_download",
        lambda _repo, filename, **_kwargs: local[filename],
    )
    monkeypatch.setattr(
        sources, "_resolve_model_for_fetch", lambda _model: ("model-commit", _D, _LAYERS),
    )
    return local


def test_rs_r_binds_both_sites_and_loads_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    from drowse.lens.store import load_lens, load_lens_sidecar

    _hub(monkeypatch, tmp_path, {
        "model/r-lens/lens.pt": _payload("R", "out"),
        "model/r-lens/mid.pt": _payload("R", "mid", scale=10.0),
    })
    binding = sources.fetch_lens_source("org/model", "rs-r")
    assert binding.provider == "rs-lenses" and binding.repo_id == "stdlogits/rs-lenses"
    assert binding.kind == "R" and binding.sites == ("out", "mid")
    assert binding.file("out").path == "model/r-lens/lens.pt"
    assert binding.file("mid").path == "model/r-lens/mid.pt"
    assert binding.corpus == "HuggingFaceFW/dclm_100BT-shuffled@abc"
    assert (binding.seq_len, binding.skip_first, binding.n_prompts) == (128, 16, 7)
    assert binding.provenance["estimator"] == "relp"
    assert binding.provenance["embedded"] is True
    assert "half_rule" in binding.provenance["rules_json"]
    on_disk = json.loads(sources.LENS_SOURCES.binding_path("org/model", "rs-r").read_text())
    assert set(on_disk["files"]) == {"R:out", "R:mid"}
    assert sources.load_external_lens_binding("org/model", "rs-r") == binding

    assert sources.LENS_SOURCES.active("org/model", "output") == SourceRef("huggingface", "rs-r")
    assert sources.LENS_SOURCES.active("org/model", "input") is None
    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None and sidecar["kind"] == "R"
    assert sidecar["sites"] == {"out": {"layers": [0, 1, 2]}, "mid": {"layers": [0, 1, 2]}}
    assert set(sidecar["_source"]["files"]) == {"R:out", "R:mid"}
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.key == LensKey("R") and lens.sites == ("out", "mid")
    assert torch.equal(lens.matrix(1, "out"), 2 * torch.eye(_D))
    assert torch.equal(lens.matrix(1, "mid"), 20 * torch.eye(_D))


def test_rs_s_with_one_site_binds_it_and_reads_never_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source may carry only one site; the other is a 404, not a substitute."""
    from drowse.lens import sources
    from drowse.lens.store import load_lens

    _hub(monkeypatch, tmp_path, {"model/s-lens/out.pt": _payload("S", "out")})
    binding = sources.fetch_lens_source("org/model", "rs-s")
    assert binding.kind == "S" and binding.sites == ("out",)
    assert set(binding.files) == {LensKey("S", "out")}
    # The S family's slot, leaving the R family's alone.
    assert sources.LENS_SOURCES.active("org/model", "input") == SourceRef("huggingface", "rs-s")
    assert sources.LENS_SOURCES.active("org/model", "output") is None
    assert load_lens("org/model") is None
    loaded = load_lens("org/model", side="input")
    assert loaded is not None
    lens, sidecar = loaded
    assert lens.kind == "S" and lens.sites == ("out",)
    assert sidecar["sites"] == {"out": {"layers": [0, 1, 2]}}
    footprint = lens.footprint(torch.ones(_D), 2, "out")
    assert torch.allclose(footprint, 3 * torch.ones(_D))
    with pytest.raises(LensNotFittedError, match="no 'mid' site"):
        lens.matrix(0, "mid")


def test_rs_files_without_a_record_bind_to_the_card_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma 4's rs-lenses files carry no provenance; the card's corpus binds."""
    from drowse.lens import sources

    _hub(monkeypatch, tmp_path, {
        "model/s-lens/out.pt": _payload("S", "out", provenance=False),
        "model/s-lens/mid.pt": _payload("S", "mid", provenance=False),
    })
    binding = sources.fetch_lens_source("org/model", "rs-s")
    assert binding.sites == ("out", "mid")
    assert binding.corpus == sources.RS_LENSES_CORPUS
    assert (binding.seq_len, binding.skip_first) == (128, 16)
    assert binding.provenance == {"estimator": "relp", "rules_json": "{}", "embedded": False}


@pytest.mark.parametrize(
    ("files", "match"),
    [
        # A file whose record names another site cannot bind as this one.
        ({"model/s-lens/out.pt": _payload("S", "mid")}, "declares lens 'S' at 'mid'"),
        # An R file in an S slot is refused by its keys.
        ({"model/s-lens/out.pt": _payload("R", "out")}, "invalid schema"),
        # A media lens (a basis beside K) is out of scope.
        ({"model/s-lens/out.pt": _payload("S", "out", basis=torch.eye(_D))}, "invalid schema"),
        # The sites of one binding are one fit.
        (
            {
                "model/s-lens/out.pt": _payload("S", "out"),
                "model/s-lens/mid.pt": _payload("S", "mid", provenance=False),
            },
            "disagree on carrying provenance",
        ),
    ],
)
def test_rs_fetch_refuses_inconsistent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, Any], match: str,
) -> None:
    from drowse.lens import sources

    _hub(monkeypatch, tmp_path, files)
    with pytest.raises(ValueError, match=match):
        sources.fetch_lens_source("org/model", "rs-s")
    assert sources.load_external_lens_binding("org/model", "rs-s") is None


def test_rs_fetch_refuses_another_models_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources

    payload = _payload("R", "out")
    payload["provenance"]["model_id"] = "org/other"
    _hub(monkeypatch, tmp_path, {"model/r-lens/lens.pt": payload})
    with pytest.raises(ValueError, match="not the requested model"):
        sources.fetch_lens_source("org/model", "rs-r")


def test_rs_fetch_names_the_published_models_on_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources

    _hub(monkeypatch, tmp_path, {
        "qwen3-8b/s-lens/out.pt": _payload("S", "out"),
        "gemma-4-12b/s-lens/mid.pt": _payload("S", "mid"),
        "gemma-4-12b/image/s-lens/mid.pt": _payload("S", "mid"),
    })
    with pytest.raises(FileNotFoundError, match="gemma-4-12b, qwen3-8b"):
        sources.fetch_lens_source("org/model", "rs-s")


def test_binding_refuses_a_site_its_provider_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources

    _hub(monkeypatch, tmp_path, {"model/r-lens/lens.pt": _payload("R", "out")})
    binding = sources.fetch_lens_source("org/model", "rs-r")
    payload = binding.to_json()
    # workspace-r publishes R:out only: an R:mid entry is not a workspace binding.
    payload.update(name="workspace-r", provider="workspace-lenses")
    payload["files"]["R:mid"] = payload["files"]["R:out"]
    payload["provenance"] = {"estimator": "relp", "rules_json": "{}", "target_layer": 2}
    write_json_atomic(sources.LENS_SOURCES.binding_path("org/model", "workspace-r"), payload)
    assert sources.load_external_lens_binding("org/model", "workspace-r") is None
    del payload["files"]["R:mid"]
    write_json_atomic(sources.LENS_SOURCES.binding_path("org/model", "workspace-r"), payload)
    assert sources.load_external_lens_binding("org/model", "workspace-r") is not None


def test_listing_names_family_and_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    from drowse.lens.jlens import JacobianLens
    from drowse.lens.store import LensProvenance, save_lens

    _hub(monkeypatch, tmp_path, {
        "model/r-lens/lens.pt": _payload("R", "out"),
        "model/s-lens/mid.pt": _payload("S", "mid"),
    })
    sources.fetch_lens_source("org/model", "rs-r")
    sources.fetch_lens_source("org/model", "rs-s")
    save_lens(
        JacobianLens(None, mid={0: torch.eye(_D)}, n_prompts=2, d_model=_D, key=LensKey("S")),
        "org/model",
        LensProvenance(corpus_spec="t", corpus_sha256="a" * 64, seq_len=8, dim_batch=1),
    )
    rows = sources.list_lens_sources("org/model")
    assert [row["source"] for row in rows] == ["rs-r", "rs-s", "local:s"]
    by = {row["source"]: row for row in rows}
    assert (by["rs-r"]["family"], by["rs-r"]["sites"], by["rs-r"]["active"]) == ("R", ["out"], True)
    assert by["rs-r"]["checkpoints"] == {"out": "model/r-lens/lens.pt"}
    assert (by["rs-s"]["family"], by["rs-s"]["sites"], by["rs-s"]["active"]) == ("S", ["mid"], False)
    assert (by["local:s"]["family"], by["local:s"]["sites"], by["local:s"]["active"]) == (
        "S", ["mid"], True,
    )


def test_compatibility_reads_every_site() -> None:
    """A mid-only source is judged on its mid layers (not an empty out)."""
    from drowse.lens.runtime import matches_loaded_model

    model = SimpleNamespace(config=SimpleNamespace(
        _commit_hash="model-commit", hidden_size=_D, num_hidden_layers=_LAYERS,
    ))
    sidecar = {
        "model_fingerprint": None,
        "d_model": _D,
        "sites": {"mid": {"layers": [0, 1, 2]}},
        "_source": {"kind": "huggingface", "model_id": "org/model", "model_revision": "model-commit"},
    }
    assert matches_loaded_model(sidecar, model, "org/model")
    sidecar["sites"] = {"mid": {"layers": [0, 3]}}
    assert not matches_loaded_model(sidecar, model, "org/model")
