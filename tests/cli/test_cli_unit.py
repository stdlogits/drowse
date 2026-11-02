"""``drowse unit`` runners: the reads through a real tiny session on CPU, the
catalog verbs without any model load."""

from __future__ import annotations

import json
from typing import Any

import pytest

from drowse.cli.main import main
from tests.support.cli import poison_session

pytestmark = pytest.mark.usefixtures("isolated_home")


def _json(capsys: pytest.CaptureFixture[str]) -> Any:
    return json.loads(capsys.readouterr().out)


def test_read_prints_one_json_document(tiny_checkpoint: Any, capsys: pytest.CaptureFixture[str]) -> None:
    model = tiny_checkpoint.model_id
    main(["unit", "read", "neuron/L1n3", "-m", model, "-d", "cpu", "-k", "3", "-j"])
    payload = _json(capsys)
    assert payload["unit"] == "neuron/L1n3" and payload["model"] == model
    assert [pole["pole"] for pole in payload["poles"]] == ["+", "-"]
    # No lens here: both halves of every place are absent with the reason.
    writes = payload["poles"][0]["writes"]
    assert writes["site"] == "out" and writes["r"]["absent"] and writes["s"]["absent"]
    assert payload["catalog"]["bound"] is False


def test_read_text_names_both_poles_and_the_catalog(
    tiny_checkpoint: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    main(["unit", "read", "head/L1h2", "-m", tiny_checkpoint.model_id, "-d", "cpu"])
    out = capsys.readouterr().out
    assert out.startswith("head/L1h2  kv head 1")
    assert "no neuron-lens catalog" in out
    for role in ("copies", "suppresses", "query", "key", "value"):
        assert role in out
    main(["unit", "read", "neuron/L2n0", "-m", tiny_checkpoint.model_id, "-d", "cpu"])
    out = capsys.readouterr().out
    assert "\n+ pole" in out and "\n- pole" in out and "fires on" in out


def test_a_bad_unit_exits_2(tiny_checkpoint: Any, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["unit", "read", "neuron/L1n9999", "-m", tiny_checkpoint.model_id, "-d", "cpu"])
    assert exc.value.code == 2
    assert "outside block 1" in capsys.readouterr().err


def test_stats_and_find(tiny_checkpoint: Any, capsys: pytest.CaptureFixture[str]) -> None:
    model = tiny_checkpoint.model_id
    main(["unit", "stats", "neuron/L1n3", "-m", model, "-d", "cpu", "-j"])
    stats = _json(capsys)
    assert stats["kind"] == "neuron" and stats["poles"][0]["firing"]["neutral"]["act_top"] >= 0
    main(["unit", "find", "neuron/L2n5", "-m", model, "-d", "cpu", "-k", "2", "-j"])
    found = _json(capsys)
    row = found["layers"]["2"]
    assert row["neurons"][0]["unit"] == "neuron/L2n5" and row["share"] == pytest.approx(1.0, abs=1e-4)


def test_catalog_verbs_never_load_a_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    attempts = poison_session(monkeypatch)
    main(["unit", "ls", "-j"])
    rows = _json(capsys)
    assert {row["model_id"] for row in rows} == {"Qwen/Qwen3-8B", "google/gemma-4-12B"}
    assert all(row["available"] and not row["bound"] for row in rows)
    main(["unit", "ls", "Qwen/Qwen3-8B"])
    assert "not fetched" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["unit", "fetch", "HuggingFaceTB/SmolLM2-360M-Instruct"])
    assert exc.value.code == 1  # 404 → 1
    assert "no neuron-lens catalog" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["unit", "rm", "Qwen/Qwen3-8B"])
    assert exc.value.code == 2 and "-y/--yes" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["unit", "rm", "Qwen/Qwen3-8B", "-y"])
    assert exc.value.code == 1 and "no unit catalog is bound" in capsys.readouterr().err
    assert attempts == []


def test_fetch_reports_the_binding(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from drowse.units import catalog as cat

    def fetch(model_id: str, *, force: bool, on_progress: Any) -> dict[str, Any]:
        on_progress("1/1 qwen3-8b/heads/labels.parquet")
        return {
            "model_id": "Qwen/Qwen3-8B", "repo": cat.CATALOG_REPO, "revision": cat.CATALOG_REVISION,
            "files": {"heads/labels.parquet": {}}, "site": "https://neuron-lens.com/qwen3-8b",
            "credit": cat.LABEL_CREDIT, "force": force,
        }

    monkeypatch.setattr(cat, "fetch_catalog", fetch)
    main(["unit", "fetch", "Qwen/Qwen3-8B", "-f"])
    captured = capsys.readouterr()
    assert "Bound stdlogits/neuron-lens@b5bf11a2b9e3" in captured.out
    assert "Labels by Claude Haiku 5.5" in captured.out
    assert "1/1 qwen3-8b/heads/labels.parquet" in captured.err
    main(["unit", "fetch", "Qwen/Qwen3-8B", "-j"])
    assert _json(capsys)["force"] is False
