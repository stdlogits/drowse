"""Lens fit checkpoints: the ten-minute rule and the graceful stop.

Regression: at 8B on MPS a document takes about four minutes, so the
25-document cadence saved once every 1.7 hours, and a SIGTERM 20 minutes into
``drowse lens fit`` left nothing.  A fit now also checkpoints at the first
document boundary ten minutes after its last checkpoint, and a graceful stop
(what the CLI's SIGTERM/SIGINT handler sets) finishes the document in flight,
checkpoints it and raises; resuming from that checkpoint is exact.
"""

from __future__ import annotations

import signal
import threading
import time
import types
from typing import Any

import pytest
import torch

import drowse.lens.fit as fit_module
from drowse.lens.fit import LensSums, fit_lens
from drowse.lens.jlens import CHECKPOINT_SECONDS, JacobianLensCancelled
from tests.support.jlens_toys import CharTokenizer, frozen_toy
from tests.support.lens import LENS_PROMPTS

_TOK = CharTokenizer()
_DOCS = [*LENS_PROMPTS, "lmkjihgfedcbamlkjihgfedcbalm"]


def _clone(sums: LensSums) -> LensSums:
    return LensSums(
        sums.kind, sums.d_model, sums.n_prompts,
        {s: {l: m.clone() for l, m in ms.items()} for s, ms in sums.total.items()},
        None if sums.odd is None else {
            s: {l: m.clone() for l, m in ms.items()} for s, ms in sums.odd.items()
        },
    )


def test_the_rule_is_ten_minutes() -> None:
    assert CHECKPOINT_SECONDS == 600.0


def test_a_fit_checkpoints_ten_minutes_after_its_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each document "takes" five minutes on a fake clock; the count cadence
    # is out of reach, so only the time rule fires: after documents 2 and 4.
    done = {"n": 0}
    fake = types.SimpleNamespace(
        monotonic=lambda: done["n"] * 300.0, perf_counter=time.perf_counter,
    )
    monkeypatch.setattr(fit_module, "time", fake)
    saved: list[int] = []
    fit_lens(
        frozen_toy(), _TOK, _DOCS, kind="R", sites=["out"], checkpoint_every=100,
        on_progress=lambda _m: done.__setitem__("n", done["n"] + 1),
        on_checkpoint=lambda sums: saved.append(sums.n_prompts),
    )
    assert saved == [2, 4]


def test_a_count_checkpoint_resets_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    done = {"n": 0}
    fake = types.SimpleNamespace(
        monotonic=lambda: done["n"] * 300.0, perf_counter=time.perf_counter,
    )
    monkeypatch.setattr(fit_module, "time", fake)
    saved: list[int] = []
    fit_lens(
        frozen_toy(), _TOK, _DOCS, kind="R", sites=["out"], checkpoint_every=3,
        on_progress=lambda _m: done.__setitem__("n", done["n"] + 1),
        on_checkpoint=lambda sums: saved.append(sums.n_prompts),
    )
    # 2: ten minutes since the start; 3: the count; 5: ten minutes since 3
    # (counting from 2 would have fired at 4).
    assert saved == [2, 3, 5]


def test_a_graceful_stop_finishes_the_document_in_flight_and_resumes_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = frozen_toy()
    full = fit_lens(model, _TOK, _DOCS, kind="R", dim_batch=2)
    stop = threading.Event()
    real_commit = fit_module._commit

    def commit(sums: LensSums, *args: Any, **kwargs: Any) -> None:
        # The stop arrives during the second document's first pass.
        if sums.n_prompts == 1:
            stop.set()
        real_commit(sums, *args, **kwargs)

    monkeypatch.setattr(fit_module, "_commit", commit)
    saved: list[LensSums] = []
    with pytest.raises(JacobianLensCancelled, match="stopped after 2 prompts"):
        fit_lens(
            model, _TOK, _DOCS, kind="R", dim_batch=2, stop_event=stop,
            on_checkpoint=lambda sums: saved.append(_clone(sums)),
        )
    assert [sums.n_prompts for sums in saved] == [2]

    monkeypatch.setattr(fit_module, "_commit", real_commit)
    resumed = fit_lens(
        model, _TOK, _DOCS[2:], kind="R", dim_batch=2, initial=saved[0], progress_base=2,
    )
    assert resumed.lens.n_prompts == len(_DOCS)
    for site in ("out", "mid"):
        for layer in full.lens.layers(site):
            assert torch.equal(resumed.lens.matrix(layer, site), full.lens.matrix(layer, site))
            assert resumed.split_half[site][layer] == full.split_half[site][layer]


def test_a_stop_right_after_a_checkpoint_does_not_write_it_twice() -> None:
    stop = threading.Event()
    saved: list[int] = []

    def keep(sums: LensSums) -> None:
        saved.append(sums.n_prompts)
        stop.set()

    with pytest.raises(JacobianLensCancelled, match="stopped after 2 prompts"):
        fit_lens(
            frozen_toy(), _TOK, _DOCS, kind="R", sites=["out"], checkpoint_every=2,
            stop_event=stop, on_checkpoint=keep,
        )
    assert saved == [2]


def test_a_stop_before_the_first_document_writes_nothing() -> None:
    stop = threading.Event()
    stop.set()
    saved: list[int] = []
    with pytest.raises(JacobianLensCancelled, match="stopped after 0 prompts"):
        fit_lens(
            frozen_toy(), _TOK, _DOCS, kind="R", sites=["out"], stop_event=stop,
            on_checkpoint=lambda sums: saved.append(sums.n_prompts),
        )
    assert saved == []


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="signal handlers install only in the main thread",
)
def test_the_cli_handler_stops_gracefully_then_at_once() -> None:
    from drowse.cli.verbs.lens import stop_on_signals

    before = signal.getsignal(signal.SIGTERM)
    with stop_on_signals() as (stop, cancel, received):
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set() and not cancel.is_set()
        signal.raise_signal(signal.SIGINT)
        assert cancel.is_set()
        assert received == [signal.SIGTERM, signal.SIGINT]
        assert signal.getsignal(signal.SIGTERM) is before  # restored after the second
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="signal handlers install only in the main thread",
)
def test_lens_fit_cli_saves_and_exits_on_sigterm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.cli.main import main
    from tests.support.cli import FakeSession, install_fake_session

    seen: dict[str, Any] = {}

    class _Lens:
        def fit(self, docs: Any, **kwargs: Any) -> None:
            signal.raise_signal(signal.SIGTERM)
            seen.update(stop=kwargs["stop_event"].is_set(), cancel=kwargs["cancel_event"].is_set())
            raise JacobianLensCancelled("lens fit stopped after 3 prompts")

    class _Session(FakeSession):
        lens = _Lens()

    install_fake_session(monkeypatch, _Session)
    corpus = tmp_path / "docs.txt"
    corpus.write_text("\n".join(_DOCS) + "\n")
    with pytest.raises(SystemExit) as exc:
        main(["lens", "fit", "m/x", "--corpus", str(corpus), "--prompts", "3"])
    assert exc.value.code == 128 + signal.SIGTERM
    assert seen == {"stop": True, "cancel": False}
    err = capsys.readouterr().err
    assert "lens fit stopped after 3 prompts. Run the same command again to resume" in err


@pytest.mark.usefixtures("isolated_home")
def test_a_stopped_lifecycle_fit_leaves_a_checkpoint_the_next_fit_resumes() -> None:
    from drowse.lens.store import load_lens_checkpoint
    from tests.support.lens import TOY_MODEL_ID, LensHarness

    stop = threading.Event()

    def on_progress(message: str) -> None:
        if message.startswith("prompt 2/"):
            stop.set()

    with pytest.raises(JacobianLensCancelled, match="stopped after 2 prompts"):
        LensHarness().lens.fit(
            LENS_PROMPTS, force=True, sites=("out",), on_progress=on_progress,
            stop_event=stop,
        )
    checkpoint = load_lens_checkpoint(TOY_MODEL_ID, "j")
    assert checkpoint is not None and checkpoint[0].n_prompts == 2

    messages: list[str] = []
    resumed = LensHarness().lens.fit(LENS_PROMPTS, sites=("out",), on_progress=messages.append)
    assert any("checkpoint at 2 prompts" in message for message in messages), messages
    assert resumed.n_prompts == len(LENS_PROMPTS)
    full = LensHarness().lens.fit(LENS_PROMPTS, force=True, sites=("out",))
    for layer in full.source_layers:
        assert torch.allclose(resumed.jacobians[layer], full.jacobians[layer], atol=1e-6)
