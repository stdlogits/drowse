"""``drowse lens``: the source listing names kind and sites, ``top`` reads
through the depth rule at a site, ``decompose`` takes a dictionary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.cli.main import main
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.read import FamilyRead, LensRead, TokenScore
from drowse.lens.store import LensProvenance, save_lens
from tests.support.cli import FakeSession, install_fake_session

_PROV = LensProvenance(corpus_spec="t", corpus_sha256="a" * 64, seq_len=8, dim_batch=1)


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "drowse"))


def _sources() -> None:
    save_lens(JacobianLens({0: torch.eye(3), 1: torch.eye(3)}, n_prompts=2, d_model=3,
                           key=LensKey("R")), "org/model", _PROV)
    save_lens(JacobianLens(None, mid={0: torch.eye(3)}, n_prompts=2, d_model=3,
                           key=LensKey("S")), "org/model", _PROV)


def test_ls_and_show_name_each_sources_kind_and_sites(capsys: pytest.CaptureFixture[str]) -> None:
    _sources()
    main(["lens", "ls", "org/model"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["* local:r  R-lens  out", "* local:s  S-lens  mid"]
    main(["lens", "ls", "org/model", "-j"])
    rows = json.loads(capsys.readouterr().out)
    assert [(row["source"], row["side"], row["sites"]) for row in rows] == [
        ("local:r", "output", ["out"]), ("local:s", "input", ["mid"]),
    ]
    main(["lens", "show", "org/model", "local:s"])
    shown = capsys.readouterr().out
    assert "S-lens local:s for org/model (S family) (active)" in shown
    assert "mid:      layers 0..0 (1)" in shown and "out:" not in shown
    main(["lens", "show", "org/model"])  # the active R-family source by default
    assert "R-lens local:r" in capsys.readouterr().out


def test_use_names_the_family_it_selected(capsys: pytest.CaptureFixture[str]) -> None:
    _sources()
    main(["lens", "use", "org/model", "local:s"])
    assert capsys.readouterr().out.strip() == "Active S-family lens for org/model: local:s"


class _Lens:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def read_prompt(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append(("read_prompt", {"prompt": prompt, **kwargs}))
        r = FamilyRead("R", 0.4, "probability", lens="R:mid", tokens=(TokenScore(5, " Paris", 0.9),))
        s = FamilyRead("S", 0.6, "footprint_z", lens="S:mid", absent="this S lens has no 'mid' site")
        return {4: [LensRead(2, "mid", 0.25, r=r, s=s)]}

    def decompose(self, selector: str, **kwargs: Any) -> Any:
        self.calls.append(("decompose", {"selector": selector, **kwargs}))
        return {2: (0.5, [(" owl", 1.0)])}


class _Session(FakeSession):
    def __init__(self, model_id: str = "fake/model", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self.lens = _Lens()


def test_top_reads_both_halves_at_the_site(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    created = install_fake_session(monkeypatch, _Session)
    main(["lens", "top", "m/x", "the prompt", "--site", "mid", "-k", "3", "--layers", "1-2"])
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "  L  2  S:mid  0.60  (absent: this S lens has no 'mid' site)",
        "        R:mid  0.40  Paris",
    ]
    (call,) = created[0].lens.calls
    assert call == ("read_prompt", {
        "prompt": "the prompt", "site": "mid", "layers": [1, 2], "positions": None, "k": 3,
    })
    main(["lens", "top", "m/x", "p", "-j"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["site"] == "out" and payload["positions"] == [4]
    (read,) = payload["reads"]["4"]
    assert read["r"]["tokens"] == [{"token_id": 5, "token": " Paris", "value": 0.9}]
    assert read["s"]["absent"] == "this S lens has no 'mid' site"


def test_decompose_takes_the_dictionary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    created = install_fake_session(monkeypatch, _Session)
    main(["lens", "decompose", "slens/owl", "-m", "m/x", "--dictionary", "tokens-in", "-j"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["dictionary"] == "tokens-in"
    assert created[0].lens.calls[0][1]["dictionary"] == "tokens-in"
    with pytest.raises(SystemExit):
        main(["lens", "decompose", "x", "-m", "m/x", "--dictionary", "neurons"])
