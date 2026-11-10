"""CPU tests for the dictionary-generic sparse lens decomposition."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from drowse.errors import WhitenerError
from drowse.geometry.whitener import LayerWhitener
from drowse.lens.decompose import TokensOut, sparse_decompose
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import LensHarness
from tests.support.whitener import isotropic_whitener, synthetic_whitener

_D = 32
_VOCAB = 60


def _iso(d: int = _D, layers: tuple[int, ...] = (0,)) -> LayerWhitener:
    """The identity metric: the whitened pursuit equals the plain one."""
    return isotropic_whitener(layers, d)


def _dictionary(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    jacobian = torch.randn(_D, _D, generator=gen) / _D**0.5
    unembed = torch.randn(_VOCAB, _D, generator=gen)
    return jacobian, unembed


def test_recovers_planted_sparse_combination() -> None:
    jacobian, unembed = _dictionary()
    atoms = {7: 3.0, 21: 2.0, 40: 1.2}
    target = torch.zeros(_D)
    for v, c in atoms.items():
        target = target + c * (unembed[v].float() @ jacobian)

    dec = sparse_decompose(target, TokensOut(jacobian, unembed, _iso(), 0), layer=0, k=8)

    got = dict(dec.atoms)
    assert set(atoms) <= set(got)
    for v, c in atoms.items():
        assert abs(got[v] - c) < 0.05, (v, got[v], c)
    assert dec.share > 0.99


def test_positive_tiny_solve_skips_projected_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = 8
    jacobian = torch.eye(d)
    unembed = torch.eye(d)
    target = 2.0 * unembed[2] + 1.5 * unembed[5]

    def _explode(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("positive NNLS case should not need eigvalsh/PGD")

    monkeypatch.setattr(torch.linalg, "eigvalsh", _explode)
    dec = sparse_decompose(target, TokensOut(jacobian, unembed, _iso(d), 0), layer=0, k=2)

    got = dict(dec.atoms)
    assert got[2] == pytest.approx(2.0, abs=1e-5)
    assert got[5] == pytest.approx(1.5, abs=1e-5)
    assert dec.share == pytest.approx(1.0)


def test_share_low_for_offspace_direction() -> None:
    jacobian, unembed = _dictionary()
    # a direction orthogonal to every atom's positive span is unreachable
    # with nonneg coefficients from a *random* dictionary only approximately;
    # instead check that pure noise explains much less than a planted target.
    gen = torch.Generator().manual_seed(3)
    noise = torch.randn(_D, generator=gen)
    dec = sparse_decompose(noise, TokensOut(jacobian, unembed, _iso(), 0), layer=0, k=4)
    assert dec.share < 0.9


def test_zero_target_yields_empty() -> None:
    jacobian, unembed = _dictionary()
    dec = sparse_decompose(torch.zeros(_D), TokensOut(jacobian, unembed, _iso(), 0), layer=0, k=4)
    assert dec.share == 0.0 and dec.atoms == []


def test_respects_k_budget() -> None:
    jacobian, unembed = _dictionary()
    gen = torch.Generator().manual_seed(5)
    target = torch.randn(_D, generator=gen)
    dec = sparse_decompose(target, TokensOut(jacobian, unembed, _iso(), 0), layer=0, k=3)
    assert len(dec.atoms) <= 3


@pytest.fixture()
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def test_lens_decompose_on_registered_profile(_isolated_home: None) -> None:
    session = LensHarness()
    session.whitener = isotropic_whitener(
        range(len(session.layers)), int(session.model.config.hidden_size),
    )
    lens = session.lens.fit(_PROMPTS)
    # register the direction for toy-vocab token 'g', then decompose it —
    # the decomposition should attribute it (near-)fully to that atom.
    name = session.lens.register_direction("g")

    out = session.lens.decompose(name, k=4)
    del lens
    assert set(out) == {0, 1}
    for share, tokens in out.values():
        assert share > 0.99
        assert tokens and tokens[0][0] == "g"


def test_tokens_out_norms_are_cached_for_every_layer(
    _isolated_home: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the tokens-out atom norms lived in an LRU sized for two
    ``[L, d, d]`` stacks, so a decomposition over more than two layers
    recomputed every layer's ``V × d²`` norm pass on every call.  One pass
    per layer now serves every later decomposition."""
    import drowse.lens.decompose as decompose_module

    session = LensHarness(n_layers=5)
    session.whitener = isotropic_whitener(
        range(len(session.layers)), int(session.model.config.hidden_size),
    )
    session.lens.fit(_PROMPTS)
    name = session.lens.register_direction("g")
    passes: list[int] = []
    real = decompose_module.token_atom_norms

    def counted(jacobian: torch.Tensor, unembed: torch.Tensor) -> torch.Tensor:
        passes.append(1)
        return real(jacobian, unembed)

    monkeypatch.setattr(decompose_module, "token_atom_norms", counted)
    first = session.lens.decompose(name, k=4)
    assert len(first) > 2  # more layers than the stack cache holds
    assert len(passes) == len(first)
    second = session.lens.decompose(name, k=4)
    assert len(passes) == len(first)  # every layer's norms were cached
    assert second == first


class _SignedDictionary:
    """A neurons-shaped dictionary: signed atoms (two poles per unit)."""

    kind = "test-signed"
    nonneg = False
    device = torch.device("cpu")

    def __init__(self, atoms: torch.Tensor) -> None:
        self._atoms = atoms

    def coords(self, vector: torch.Tensor) -> torch.Tensor:
        return vector.float()

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        return self._atoms @ residual

    def atom(self, index: int) -> torch.Tensor:
        return self._atoms[index]

    def norms(self) -> torch.Tensor:
        return self._atoms.norm(dim=-1)


def test_signed_dictionary_recovers_negative_coefficients() -> None:
    """The dictionary hook carries signed atoms (the future ``neurons``)."""
    gen = torch.Generator().manual_seed(11)
    atoms = torch.randn(_VOCAB, _D, generator=gen)
    target = 2.0 * atoms[4] - 1.5 * atoms[17]

    dec = sparse_decompose(target, _SignedDictionary(atoms), layer=3, k=4)

    got = dict(dec.atoms)
    assert got[4] == pytest.approx(2.0, abs=1e-3)
    assert got[17] == pytest.approx(-1.5, abs=1e-3)
    assert dec.share == pytest.approx(1.0, abs=1e-5)
    assert dec.dictionary == "test-signed" and dec.layer == 3


def test_token_dictionary_matches_the_composed_lens_matvec() -> None:
    jacobian, unembed = _dictionary()
    words = TokensOut(jacobian, unembed, _iso(), 0)
    residual = torch.randn(_D, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(words.scores(residual), unembed @ (jacobian @ residual))
    assert torch.allclose(words.atom(9), unembed[9] @ jacobian)
    assert torch.allclose(words.norms(), (unembed @ jacobian).norm(dim=-1), atol=1e-5)


def test_the_pursuit_runs_in_the_whitened_metric() -> None:
    """Regression: selection, the NNLS fit and the share used Euclidean
    residual-space products.  Under an anisotropic covariance they are the
    Mahalanobis ones: equal to the Euclidean pursuit of the whitened target
    (``Σ^{-1/2}t``, a vector) against the dual-whitened atoms (``Σ^{1/2}u``,
    token atoms are reads), and different from the plain one."""
    jacobian, unembed = _dictionary()
    whitener = synthetic_whitener([0], _D)
    gen = torch.Generator().manual_seed(9)
    target = torch.randn(_D, generator=gen)

    dec = sparse_decompose(target, TokensOut(jacobian, unembed, whitener, 0), layer=0, k=4)
    # By hand: whiten the target as a vector and the atoms as reads, then
    # the identity-metric pursuit.
    W = whitener.whiten(0, torch.eye(_D))       # Σ^{-1/2} (symmetric)
    W_dual = whitener.whiten_dual(0, torch.eye(_D))  # Σ^{+1/2}
    hand = sparse_decompose(
        target @ W, TokensOut(jacobian @ W_dual, unembed, _iso(), 0), layer=0, k=4,
    )
    assert [v for v, _c in dec.atoms] == [v for v, _c in hand.atoms]
    assert dec.share == pytest.approx(hand.share, rel=1e-4)
    plain = sparse_decompose(target, TokensOut(jacobian, unembed, _iso(), 0), layer=0, k=4)
    assert dec.atoms != plain.atoms or abs(dec.share - plain.share) > 1e-3
    # The whitened coordinates' dot products are Mahalanobis, and a read
    # pairs with a vector exactly as before whitening.
    u, v = torch.randn(2, _D, generator=gen)
    assert float(whitener.whiten(0, u) @ whitener.whiten(0, v)) == pytest.approx(
        float(u @ whitener.apply_inv(0, v)), rel=1e-4,
    )
    assert float(whitener.whiten_dual(0, u) @ whitener.whiten(0, v)) == pytest.approx(
        float(u @ v), rel=1e-4,
    )


def test_an_uncovered_layer_is_a_whitener_error() -> None:
    jacobian, unembed = _dictionary()
    with pytest.raises(WhitenerError, match="layer 3"):
        TokensOut(jacobian, unembed, _iso(layers=(0,)), 3)
    session = LensHarness()
    session.lens.fit(_PROMPTS)
    # A lens atom is a read; raising it to a steering direction needs Σ.
    with pytest.raises(WhitenerError, match="needs a Mahalanobis whitener"):
        session.lens.register_direction("g")
    width = int(session.model.config.hidden_size)
    session.registry.put_profile(
        "local/v", {L: torch.ones(width) for L in range(len(session.layers))},
    )
    with pytest.raises(WhitenerError, match="none could be built"):
        session.lens.decompose("local/v", k=2)
