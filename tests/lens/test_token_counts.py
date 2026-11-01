"""The S read's token mask from token counts: the rule, the corpus-counts cache,
the source order and the session's rebuild (decision 2, Wave 4)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.errors import RequestError
from drowse.lens import token_counts as tc
from drowse.lens.footprint import (
    MIN_TOKEN_COUNT,
    TOKEN_COUNT_BASIS,
    TokenCounts,
    TokenEmbeddings,
    count_threshold,
    mask_detail,
    token_mask,
)
from tests.lens.test_lens_footprint import V, _Tok, _weight

pytestmark = pytest.mark.usefixtures("isolated_home")


def _counts(values: list[int], source: str = "catalog", min_count: int = MIN_TOKEN_COUNT) -> TokenCounts:
    return TokenCounts(
        counts=torch.tensor(values, dtype=torch.int64), source=source,  # type: ignore[arg-type]
        detail="test counts", min_count=min_count,
    )


def test_counts_drop_rare_tokens_on_top_of_the_tokenizer_rules() -> None:
    """Regression (Qwen3-8B): tokens neuron-lens saw under 5 times in 88M
    (clarsimp, ' yytype', 邓小) filled 94 of 112 L2 "fires on" S tokens; the
    mask now also drops every id counted under the threshold, and an id past
    the counts (never counted) too."""
    base = token_mask(_Tok(), V)
    # Ten counted ids (the tokenizer's length): 1 and 6 seen under 5 times.
    counted = token_mask(_Tok(), V, _counts([9, 2, 9, 9, 9, 9, 4, 9, 9, 9]))
    assert counted.tolist() == [
        True, False, False, False, False, True, False, False, True, False, False, False,
    ]
    assert (counted & ~base).sum() == 0, "counts only ever drop"
    # Counts shorter than the vocabulary: the uncounted ids are dropped.
    short = token_mask(_Tok(), V, _counts([9, 9, 9, 9, 9, 9]))
    assert short[6:].sum() == 0 and short[:6].tolist() == base[:6].tolist()
    # The threshold is the counts' own.
    assert token_mask(_Tok(), V, _counts([9, 2, 9, 9, 9, 9, 4, 9, 9, 9], min_count=2))[1]


def test_the_corpus_threshold_keeps_neuron_lens_rate() -> None:
    assert count_threshold(TOKEN_COUNT_BASIS) == MIN_TOKEN_COUNT
    assert count_threshold(16_000_000) == 1
    assert count_threshold(40_000_000) == 3
    assert count_threshold(10) == 1
    source, detail = mask_detail(None)
    assert source == "tokenizer" and "drowse lens counts" in detail and "drowse unit fetch" in detail
    assert mask_detail(_counts([1], min_count=5)) == ("catalog", "test counts, kept at 5+")


def test_embedding_statistics_name_their_mask() -> None:
    counts = _counts([9, 2, 9, 9, 9, 9, 4, 9, 9, 9])
    keep = token_mask(_Tok(), V, counts)
    emb = TokenEmbeddings.build(_weight().float(), 1.0, keep, counts=counts)
    assert emb.describe() == {"source": "catalog", "detail": "test counts, kept at 5+", "kept": int(keep.sum())}
    bare = TokenEmbeddings.build(_weight().float(), 1.0, token_mask(_Tok(), V))
    assert bare.mask_source == "tokenizer"


class _WordTokenizer:
    """Whitespace words over a fixed vocabulary (``len`` is its size)."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = {word: i for i, word in enumerate(vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def __call__(self, texts: list[str], add_special_tokens: bool = False) -> dict[str, Any]:
        assert add_special_tokens is False
        return {"input_ids": [[self.vocab[w] for w in text.split()] for text in texts]}


def test_corpus_counts_are_a_model_level_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = _WordTokenizer(["a", "b", "c", "d"])
    counts, docs, tokens = tc.count_tokens(tok, ["a a b", "", "c a"], batch=1)
    assert counts.tolist() == [3, 1, 1, 0] and docs == 2 and tokens == 5
    with pytest.raises(RequestError, match="at least"):
        tc.write_corpus_counts("org/m", counts, corpus="x", documents=docs, tokens=tokens, tokenizer="t")
    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    record = tc.write_corpus_counts(
        "org/m", counts, corpus="repo@abc", documents=docs, tokens=40_000_000, tokenizer="t",
    )
    assert record["min_count"] == 3 and record["vocabulary"] == 4
    assert tc.counts_dir("org/m").name == "tokens"
    loaded = tc.load_corpus_counts("org/m")
    assert loaded is not None and loaded.source == "corpus" and loaded.min_count == 3
    assert loaded.counts.tolist() == [3, 1, 1, 0]
    assert "40,000,000 tokens of repo@abc" in loaded.detail
    # Another format is not read; nor another model's record.
    path = tc.counts_dir("org/m") / "counts.json"
    stale = {**json.loads(path.read_text()), "format": 0}
    path.write_text(json.dumps(stale))
    assert tc.load_corpus_counts("org/m") is None
    assert tc.remove_corpus_counts("org/m") and not tc.remove_corpus_counts("org/m")


def test_the_source_prefers_the_catalog_and_reloads_on_disk_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    catalog: dict[str, Any] = {"value": None}
    source = tc.TokenCountSource("org/m", lambda: catalog["value"])
    assert source.get() is None and source.describe()["source"] == "tokenizer"
    tc.write_corpus_counts(
        "org/m", torch.tensor([5, 0, 7]), corpus="c", documents=1, tokens=3, tokenizer="t",
    )
    first = source.get()
    assert first is not None and first.source == "corpus"
    assert source.get() is first, "unchanged on disk: the same object (a cache key)"
    tc.write_corpus_counts(
        "org/m", torch.tensor([5, 1, 7]), corpus="c", documents=1, tokens=3, tokenizer="t",
    )
    second = source.get()
    assert second is not None and second is not first and second.counts.tolist() == [5, 1, 7]
    bound = _counts([9, 9, 9])
    catalog["value"] = SimpleNamespace(token_counts=bound)
    assert source.get() is bound and source.describe()["source"] == "catalog"
    catalog["value"] = SimpleNamespace(token_counts=None)   # a binding without counts
    assert source.get() is second


def test_counts_of_another_tokenizer_are_not_read(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: corpus counts were applied to any tokenizer of the model id
    (the record's tokenizer was written and never compared), so counts taken
    over another vocabulary masked the wrong ids."""
    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    session = make_tiny_session()
    vocab = len(session.tokenizer)
    counts = torch.full((vocab,), 3, dtype=torch.int64)
    tc.write_corpus_counts(
        session.model_id, counts, corpus="file:test", documents=1, tokens=5_000_000,
        tokenizer="999:0123456789abcdef",
    )
    with caplog.at_level("WARNING", logger="drowse.lens.token_counts"):
        assert session.lens.token_mask_info()["source"] == "tokenizer"
    assert "another tokenizer" in caplog.text
    tc.write_corpus_counts(
        session.model_id, counts, corpus="file:test", documents=1, tokens=5_000_000,
        tokenizer=tc.tokenizer_fingerprint(session.tokenizer),
    )
    assert session.lens.token_mask_info()["source"] == "corpus"
    # The fingerprint is the vocabulary: its size and every (piece, id) pair.
    fp = tc.tokenizer_fingerprint(session.tokenizer)
    assert fp.startswith(f"{vocab}:")
    assert tc.tokenizer_fingerprint(_WordTokenizer(["a", "b"])) != tc.tokenizer_fingerprint(_WordTokenizer(["b", "a"]))


def test_the_session_masks_by_corpus_counts_and_rebuilds_on_a_change(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``lens counts`` beside a running session reaches its next S read, and
    a unit read's kept-token cache follows the rebuilt mask."""
    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    session = make_tiny_session()
    before = session.lens.token_embeddings()
    assert before.mask_source == "tokenizer"
    assert session.lens.token_mask_info() == {
        "source": "tokenizer", "detail": mask_detail(None)[1], "kept": before.n_kept,
    }
    vocab = len(session.tokenizer)
    counts = torch.full((vocab,), 3, dtype=torch.int64)
    kept_ids = before.keep.nonzero().squeeze(1).tolist()
    rare = kept_ids[:4]
    counts[rare] = 0
    tc.write_corpus_counts(
        session.model_id, counts, corpus="file:test", documents=1, tokens=5_000_000,
        tokenizer=tc.tokenizer_fingerprint(session.tokenizer),
    )
    info = session.lens.token_mask_info()
    assert info["source"] == "corpus" and info["kept"] is None, "counted since the last build"
    after = session.lens.token_embeddings()
    assert after is not before and after.mask_source == "corpus"
    assert after.n_kept == before.n_kept - len(rare)
    assert not any(bool(after.keep[i]) for i in rare)
    assert session.lens.token_embeddings() is after, "unchanged: no rebuild"
    reader = session.units.reads
    kept = reader._kept()
    assert int(kept.numel()) == after.n_kept
    tc.remove_corpus_counts(session.model_id)
    assert session.lens.token_embeddings().mask_source == "tokenizer"
    assert int(reader._kept().numel()) == before.n_kept


def test_a_mask_change_refreshes_every_s_view(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a token-count change rebuilt the embedding statistics but
    left the S family's other views on the old mask: a registered ``slens/``
    direction returned early (its old centring, even for a token the new mask
    drops), an attached ``slens/`` probe read the old statistics until another
    S read rebuilt them, and the probe digest did not name the mask."""
    from drowse.generation.sampling import SamplingConfig
    from drowse.lens.jlens import InvalidTokenIdError
    from tests.lens.test_slens import _install_s_lens, _token

    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    session = make_tiny_session()
    _install_s_lens(session)
    a, b = _token(session, "a"), _token(session, "b")
    session.add_probe("slens/a")
    session.add_probe("slens/b")
    session.registry.ensure_profile_registered("slens/a")
    session.registry.ensure_profile_registered("slens/b")
    before = session.lens.token_embeddings()
    old_a = session.registry.profiles["slens/a"]
    old_hash = session.probe_hashes()["slens/a"]

    counts = torch.full((len(session.tokenizer),), 3, dtype=torch.int64)
    rare = [i for i in before.keep.nonzero().squeeze(1).tolist() if i not in (a, b)][:4]
    counts[rare + [b]] = 0
    tc.write_corpus_counts(
        session.model_id, counts, corpus="file:test", documents=1, tokens=5_000_000,
        tokenizer=tc.tokenizer_fingerprint(session.tokenizer),
    )
    # The instrument's digest names the mask the next run reads (disk only).
    assert session.lens.probe_hash("slens/a") != old_hash
    # The next run rebuilds the statistics before it binds: a probe whose token
    # the new mask drops is detached, the others read the new mask.
    session.generate("hi", sampling=SamplingConfig(max_tokens=2), stateless=True)
    after = session.lens.token_embeddings()
    assert after is not before and after.mask_source == "corpus"
    assert "slens/b" not in session.lens.probes and "slens/a" in session.lens.probes
    new_hash = session.probe_hashes()["slens/a"]
    assert new_hash != old_hash
    # Registered directions re-resolve over the new centring and re-validate.
    assert "slens/a" not in session.registry.profiles
    session.registry.ensure_profile_registered("slens/a")
    assert not torch.equal(session.registry.profiles["slens/a"][1].cpu(), old_a[1].cpu())
    with pytest.raises(InvalidTokenIdError, match="mask"):
        session.registry.ensure_profile_registered("slens/b")


def test_lens_counts_counts_a_corpus_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import transformers

    from drowse.cli.main import main

    monkeypatch.setattr(tc, "MIN_COUNT_TOKENS", 1)
    tok = _WordTokenizer(["a", "b", "c"])
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, model, **kw: tok),
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b a\nc a\n")
    main(["lens", "counts", "org/m", "--corpus", str(corpus), "-j"])
    record = json.loads(capsys.readouterr().out)
    assert record["tokens"] == 5 and record["documents"] == 2 and record["corpus"] == "file:corpus.txt"
    assert tc.load_corpus_counts("org/m") is not None
    main(["lens", "counts", "org/m", "--corpus", str(corpus)])
    assert "Already counted" in capsys.readouterr().out
    main(["lens", "counts", "org/m", "--remove"])
    assert tc.load_corpus_counts("org/m") is None


def test_a_recorded_corpus_spec_streams_again() -> None:
    """``lens counts`` streams the S lens's own corpus: a local fit's
    ``hf:…`` / ``mix:…`` spec and an import's plain ``REPO@COMMIT`` read back
    to the corpus; a file corpus (or anything else) does not."""
    from drowse.lens.corpus import HfLensCorpus, LensCorpusMix, corpus_from_spec

    sha = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    one = HfLensCorpus("HuggingFaceFW/fineweb-edu", "sample-10BT", sha)
    spread = HfLensCorpus("HuggingFaceFW/fineweb-2", "cmn_Hani", sha, "content", 64, 3)
    assert corpus_from_spec(one.spec) == one
    assert corpus_from_spec(spread.spec) == spread
    mix = LensCorpusMix((one, spread), (3, 1))
    assert corpus_from_spec(mix.spec) == mix
    imported = corpus_from_spec("HuggingFaceFW/dclm_100BT-shuffled@2fa015e4044ec442a0734e89658cdcc538d10dd4")
    assert imported == HfLensCorpus(
        "HuggingFaceFW/dclm_100BT-shuffled", None, "2fa015e4044ec442a0734e89658cdcc538d10dd4",
    )
    for unreadable in ("file:corpus.txt", "hf:broken", "HuggingFaceFW/dclm_100BT-shuffled", "mix:x*y"):
        assert corpus_from_spec(unreadable) is None, unreadable
