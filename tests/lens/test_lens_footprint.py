"""Footprint z, the token mask, the four-lens read and the tokens-in dictionary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.lens.decompose import TokensIn, sparse_decompose
from drowse.lens.depth import LensBand, kind_weights
from drowse.lens.footprint import TokenEmbeddings, pullback_rows, token_mask
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.read import read_vectors
from tests.support.whitener import isotropic_whitener

V, D = 12, 5


class _Tok:
    """A tokenizer whose id 3 is special, 9 an added special token, 4 an
    added control token not marked special, 7 a partial UTF-8 byte, 2 the
    sentencepiece byte piece ``<0x41>`` (which decodes to plain ``A``), and
    whose vocabulary stops two rows short of the embedding."""

    all_special_ids = [3]
    added_tokens_decoder = {9: SimpleNamespace(special=True), 4: SimpleNamespace(special=False)}

    def __len__(self) -> int:
        return V - 2

    def batch_decode(self, rows: list[list[int]]) -> list[str]:
        return ["�" if row == [7] else "A" if row == [2] else f"t{row[0]}" for row in rows]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return ["<0x41>" if i == 2 else "<0xE2>" if i == 7 else f"t{i}" for i in ids]


def _weight(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(V, D, generator=gen, dtype=torch.float64)


def test_token_mask_drops_special_added_byte_fallback_and_padding_rows() -> None:
    """Regression: ASCII byte pieces (``<0x00>``..``<0x7F>``, half of Gemma
    4's byte fallback) decode to valid characters, so only their spelling
    marks them; added control tokens not marked special (Qwen3's
    ``<think>``) are dropped with the special ones."""
    keep = token_mask(_Tok(), V)
    assert keep.tolist() == [
        True, True, False, False, False, True, True, False, True, False, False, False,
    ]


def test_token_mask_without_a_spelling_falls_back_to_decoding() -> None:
    class _Bare:
        all_special_ids: list[int] = []

        def __len__(self) -> int:
            return 3

        def decode(self, ids: list[int]) -> str:
            return "�" if ids == [1] else "x"

    assert token_mask(_Bare(), 4).tolist() == [True, False, True, False]


def test_embedding_statistics_are_the_masked_scaled_moments() -> None:
    weight = _weight()
    keep = token_mask(_Tok(), V)
    emb = TokenEmbeddings.build(weight, 2.0, keep, chunk=4)
    rows = weight[keep].float() * 2.0
    assert emb.n_kept == int(keep.sum())
    assert torch.allclose(emb.mean, rows.mean(0), atol=1e-6)
    assert torch.allclose(emb.cov, torch.cov(rows.T, correction=0), atol=1e-5)
    assert emb.mean.dtype == emb.cov.dtype == torch.float32


def test_footprint_z_is_the_vocabulary_z_score_and_ranks_like_neuron_lens() -> None:
    weight = _weight(1)
    keep = token_mask(_Tok(), V)
    emb = TokenEmbeddings.build(weight, 3.0, keep)
    u = torch.randn(4, D, generator=torch.Generator().manual_seed(2))
    z = emb.z(u)
    assert torch.isneginf(z[:, ~keep]).all()
    kept = z[:, keep]
    assert torch.allclose(kept.mean(1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(kept.std(1, correction=0), torch.ones(4), atol=1e-5)
    # neuron-lens's read: a unit covector scored against the centred kept rows.
    centred = weight[keep].float() * 3.0
    centred -= centred.mean(0)
    raw = (u / u.norm(dim=1, keepdim=True)) @ centred.T
    kept_ids = keep.nonzero().squeeze(1)
    assert torch.equal(kept_ids[raw.topk(4).indices], z.topk(4).indices)
    # The selected-token path agrees with the full one.
    ids = torch.tensor([0, 5, 8])
    assert torch.allclose(emb.z_of(u, ids), z[:, ids], atol=1e-5)
    # A zero read is z = 0, not NaN.
    assert torch.equal(emb.z(torch.zeros(1, D))[:, keep], torch.zeros(1, int(keep.sum())))


def test_pullback_rows_is_k_transpose_h() -> None:
    gen = torch.Generator().manual_seed(3)
    K = torch.randn(2, D, D, generator=gen)
    h = torch.randn(2, D, generator=gen)
    expected = torch.stack([K[i].T @ h[i] for i in range(2)])
    assert torch.allclose(pullback_rows(K, h), expected, atol=1e-6)


def _read(**kwargs: Any) -> Any:
    weight = _weight(4)
    emb = TokenEmbeddings.build(weight, 1.0, token_mask(_Tok(), V))
    head = torch.nn.Linear(D, V, bias=False).double()
    defaults: dict[str, Any] = dict(
        readout=lambda: (lambda x: head(x.double()).float()),
        embeddings=lambda: emb,
        decode=lambda i: f"t{i}",
        n_layers=11,
        device=torch.device("cpu"),
        k=3,
    )
    defaults.update(kwargs)
    return head, emb, read_vectors(**defaults)


def test_read_returns_both_halves_weighted_by_depth() -> None:
    gen = torch.Generator().manual_seed(5)
    J = {L: torch.randn(D, D, generator=gen) for L in range(11)}
    K = {L: torch.randn(D, D, generator=gen) for L in range(11)}
    r_lens = JacobianLens(None, mid=J, n_prompts=1, d_model=D, key=LensKey("R"))
    s_lens = JacobianLens(None, mid=K, n_prompts=1, d_model=D, key=LensKey("S"))
    x = torch.randn(2, D, generator=gen)
    head, emb, reads = _read(
        vectors=x, layer=3, site="mid", r_lens=r_lens, s_lens=s_lens,
        band=LensBand(0.2, 0.4),
    )
    assert len(reads) == 2
    w_s, w_r = kind_weights(11, LensBand(0.2, 0.4))[3]
    for row, read in enumerate(reads):
        assert (read.layer, read.site, read.depth) == (3, "mid", 0.3)
        assert (read.r.weight, read.s.weight) == (w_r, w_s) and read.r.weight + read.s.weight == 1
        assert (read.r.lens, read.s.lens) == ("R:mid", "S:mid")
        assert read.r.unit == "probability" and read.s.unit == "footprint_z"
        probs = head((x[row] @ J[3].T).double()).softmax(-1)
        top = probs.topk(3)
        assert [t.token_id for t in read.r.tokens] == top.indices.tolist()
        assert [t.value for t in read.r.tokens] == pytest.approx(top.values.tolist(), abs=1e-5)
        z = emb.z((x[row] @ K[3]).reshape(1, -1))[0]
        assert [t.token_id for t in read.s.tokens] == z.topk(3).indices.tolist()
        assert read.s.tokens[0].token == f"t{read.s.tokens[0].token_id}"
    payload = reads[0].to_dict()
    assert set(payload) == {"layer", "site", "depth", "r", "s"}
    assert set(payload["s"]) == {"family", "weight", "unit", "lens", "tokens", "absent"}


def test_an_absent_half_says_why_and_never_reads_the_other_site() -> None:
    eye = {L: torch.eye(D) for L in range(4)}
    r_out_only = JacobianLens(eye, n_prompts=1, d_model=D, key=LensKey("R"))
    calls: list[str] = []

    def readout() -> Any:
        calls.append("readout")
        return lambda x: x

    def embeddings() -> Any:
        calls.append("embeddings")
        raise AssertionError("an absent S half never builds the embedding statistics")

    _head, _emb, (read,) = _read(
        vectors=torch.ones(D), layer=1, site="mid", r_lens=r_out_only, s_lens=None,
        readout=readout, embeddings=embeddings, n_layers=4,
        no_s_lens="no S lens: fetch rs-s",
    )
    assert not read.r.present and read.r.lens == "R:mid" and read.r.tokens == ()
    assert "no 'mid' site" in str(read.r.absent)
    assert not read.s.present and read.s.lens is None and read.s.absent == "no S lens: fetch rs-s"
    assert calls == []
    # A layer the site lacks is absent too, by name.
    _head, _emb, (read,) = _read(
        vectors=torch.ones(D), layer=3, site="out",
        r_lens=JacobianLens({0: torch.eye(D)}, n_prompts=1, d_model=D, key=LensKey("R")),
        s_lens=None, n_layers=4,
    )
    assert "layer 3 not in fitted R:out" in str(read.r.absent)


def test_read_refuses_bad_places_and_crossed_families() -> None:
    lens = JacobianLens({0: torch.eye(D)}, n_prompts=1, d_model=D, key=LensKey("S"))
    with pytest.raises(ValueError, match="site must be"):
        _read(vectors=torch.ones(D), layer=0, site="pre", r_lens=None, s_lens=None)
    with pytest.raises(ValueError, match="outside the model"):
        _read(vectors=torch.ones(D), layer=11, site="out", r_lens=None, s_lens=None)
    with pytest.raises(ValueError, match="cannot read the output-family half"):
        _read(vectors=torch.ones(D), layer=0, site="out", r_lens=lens, s_lens=None)


def test_tokens_in_recovers_a_mix_of_footprints() -> None:
    weight = _weight(6)
    keep = token_mask(_Tok(), V)
    emb = TokenEmbeddings.build(weight, 1.0, keep)
    K = torch.randn(D, D, generator=torch.Generator().manual_seed(7)) + 3 * torch.eye(D)
    target = 2.0 * (K @ emb.centred([1])[0]) + 0.5 * (K @ emb.centred([6])[0])
    dictionary = TokensIn(K, emb, isotropic_whitener([0], D), 0)
    assert dictionary.kind == "tokens-in" and dictionary.nonneg
    norms = dictionary.norms()
    assert norms[3] == 1.0 and torch.isneginf(dictionary.scores(target)[~keep]).all()
    assert torch.allclose(norms[1], (K @ emb.centred([1])[0]).norm())
    dec = sparse_decompose(target, dictionary, layer=0, k=2)
    assert dec.dictionary == "tokens-in"
    assert sorted(idx for idx, _ in dec.atoms) == [1, 6]
    assert dec.share == pytest.approx(1.0, abs=1e-4)


def test_tokens_in_selects_and_shares_in_the_whitened_metric() -> None:
    """Regression (review repro): with ``K = I``, centred embeddings
    ``±(10, 1)`` and ``±(0, 1)``, target ``(1, 1)`` and covariance
    ``diag(100, 1)``, the Euclidean pursuit picked token 0 at share 0.599;
    the Mahalanobis one picks token 2 at share 0.990."""
    from drowse.geometry.whitener import LayerWhitener

    weight = torch.tensor([[10.0, 1.0], [-10.0, -1.0], [0.0, 1.0], [0.0, -1.0]])
    emb = TokenEmbeddings.build(weight, 1.0, torch.ones(4, dtype=torch.bool))
    # Σ_reg = λI + XᵀX/N = I + diag(99, 0) = diag(100, 1).
    whitener = LayerWhitener(
        {0: torch.tensor([[99.0**0.5, 0.0]])}, {0: torch.tensor([[0.01]])},
        {0: 1.0}, {0: torch.zeros(2)},
    )
    dec = sparse_decompose(
        torch.tensor([1.0, 1.0]), TokensIn(torch.eye(2), emb, whitener, 0), layer=0, k=1,
    )
    assert [idx for idx, _ in dec.atoms] == [2]
    assert dec.share == pytest.approx(1.0 / 1.01, rel=1e-4)
