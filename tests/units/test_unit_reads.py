"""Static unit reads against hand computations on a tiny session.

The session carries hand-built R and S lenses at both sites (identity plus
noise per layer; the R ``out`` site stops a layer short of the last block, as
a fitted one does), so every half is a known matrix product.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from drowse.lens.depth import kind_weights
from drowse.lens.footprint import token_mask
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.model.blocks import model_spec
from drowse.model.units import (
    Pole, head_read_rows, head_write_matrix, neuron_read, neuron_read_rows, neuron_reads, neuron_write,
    neuron_writes,
)
from drowse.naming import HeadRef, NeuronRef
from drowse.units.reads import (
    HeadRead, NeuronRead, PlaceRead, copy_score, full_logits, logit_moments, own_token_z, token_logits,
)

pytestmark = pytest.mark.usefixtures("isolated_home")


def _install_lens(session: Any, kind: str, *, seed: int, last_out: bool) -> JacobianLens:
    """Publish (and select) a local ``kind`` lens at both sites."""
    from drowse.lens.store import LensProvenance, save_lens
    from drowse.model.fingerprint import loaded_model_fingerprint

    n_layers = len(session.layers)
    d = int(session.model.get_input_embeddings().weight.shape[1])
    gen = torch.Generator().manual_seed(seed)
    mid = {L: torch.eye(d) + 0.2 * torch.randn(d, d, generator=gen) for L in range(n_layers)}
    out_layers = range(n_layers) if last_out else range(n_layers - 1)
    out = {L: torch.eye(d) + 0.2 * torch.randn(d, d, generator=gen) for L in out_layers}
    lens = JacobianLens(out, mid=mid, n_prompts=3, d_model=d, key=LensKey(kind))
    save_lens(lens, session.model_id, LensProvenance(
        corpus_spec="test", corpus_sha256="a" * 64, seq_len=8, dim_batch=1,
        model_fingerprint=loaded_model_fingerprint(session.model, session.model_id),
    ))
    return lens


@pytest.fixture
def lensed(make_tiny_session: Any) -> tuple[Any, JacobianLens, JacobianLens]:
    session = make_tiny_session()
    r = _install_lens(session, "R", seed=1, last_out=False)
    s = _install_lens(session, "S", seed=2, last_out=True)
    session.lens.select_source("local:r")
    session.lens.select_source("local:s")
    return session, r, s


def _readout(session: Any) -> Any:
    return session.lens.runtime.readout_modules()[1]


def _keep(session: Any) -> torch.Tensor:
    weight = session.model.get_input_embeddings().weight
    return token_mask(session.tokenizer, int(weight.shape[0]))


def _centred(session: Any) -> torch.Tensor:
    """``ẽ`` for every token (llama's embedding scale is 1), centred over the mask."""
    weight = session.model.get_input_embeddings().weight.detach().float()
    return weight - weight[_keep(session)].mean(0)


def _r_probs(session: Any, rows: torch.Tensor, J: torch.Tensor | None) -> torch.Tensor:
    x = rows.float() if J is None else rows.float() @ J.T
    return _readout(session)(x).float().softmax(-1)


def _footprint_z(session: Any, rows: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    keep = _keep(session)
    u = rows.float() @ K
    scores = u @ _centred(session).T
    std = scores[:, keep].std(dim=-1, correction=0)
    return (scores / std[:, None]).masked_fill(~keep, -torch.inf)


def _ids(half: Any) -> list[int]:
    return [t.token_id for t in half.tokens]


# -- neurons -------------------------------------------------------------------


def test_neuron_writes_and_fires_on_by_hand(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    L, n = 1, 5
    block = spec.blocks[L]
    read = cast(NeuronRead, session.units.read(f"neuron/L{L}n{n}", k=4))
    assert read.ref == NeuronRef(layer=L, index=n)
    w = neuron_write(block, n)
    w_s, w_r = kind_weights(len(session.layers), session.lens.band)[L]
    poles: tuple[tuple[Pole, float], ...] = (("+", 1.0), ("-", -1.0))
    for pole, sign in poles:
        place = read.poles[0 if sign > 0 else 1]
        assert place.pole == pole
        writes = place.writes
        assert (writes.layer, writes.site, writes.hubs) == (L, "out", "shown")
        assert (writes.s.weight, writes.r.weight) == pytest.approx((w_s, w_r))
        assert (writes.r.lens, writes.s.lens) == ("R:out", "S:out")
        probs = _r_probs(session, (sign * w)[None], r.matrix(L, "out"))[0]
        assert _ids(writes.r) == probs.topk(4).indices.tolist()
        assert [t.value for t in writes.r.tokens] == pytest.approx(probs.topk(4).values.tolist(), rel=1e-4)
        z = _footprint_z(session, (sign * w)[None], s.matrix(L, "out"))[0]
        assert _ids(writes.s) == z.topk(4).indices.tolist()
        assert [t.value for t in writes.s.tokens] == pytest.approx(z.topk(4).values.tolist(), rel=1e-4)

        c = neuron_read(block, n, pole)
        fires = place.fires_on
        assert (fires.layer, fires.site) == (L, "mid")
        assert (fires.r.lens, fires.s.lens) == ("R:mid", "S:mid")
        probs = _r_probs(session, c[None], r.matrix(L, "mid"))[0]
        assert _ids(fires.r) == probs.topk(4).indices.tolist()
        z = _footprint_z(session, c[None], s.matrix(L, "mid"))[0]
        assert _ids(fires.s) == z.topk(4).indices.tolist()


def test_unit_rows_take_the_lens_direction_read(lensed: Any) -> None:
    """A unit's rows are weight directions: their raw halves are the lens's
    uncentred direction read, never a residual read centred on the neutral
    mean at the site."""
    session, _r, _s = lensed
    spec = model_spec(session.model)
    L, n = 1, 5
    read = cast(NeuronRead, session.units.read(f"neuron/L{L}n{n}", k=4))
    w = neuron_write(spec.blocks[L], n)
    direction = session.lens.read(w, L, "out", of="direction", k=4)
    residual = session.lens.read(w, L, "out", of="residual", k=4)
    assert not direction.s.centred and residual.s.centred
    writes = read.poles[0].writes
    assert _ids(writes.s) == [t.token_id for t in direction.s.tokens]
    assert [t.value for t in writes.s.tokens] == pytest.approx(
        [t.value for t in direction.s.tokens], rel=1e-5,
    )
    assert [t.value for t in writes.s.tokens] != pytest.approx(
        [t.value for t in residual.s.tokens], rel=1e-3,
    )


def test_neuron_scalars_by_hand(lensed: Any) -> None:
    session, _r, _s = lensed
    spec = model_spec(session.model)
    L, n = 2, 9
    block = spec.blocks[L]
    read = cast(NeuronRead, session.units.read(f"neuron/L{L}n{n}"))
    w = neuron_write(block, n)
    a = block.mlp.gate.weight.detach().float()[n] * block.mlp_in_norm.gain()
    b = block.mlp.up.weight.detach().float()[n] * block.mlp_in_norm.gain()
    assert read.w_norm == pytest.approx(float(w.norm()), rel=1e-5)
    assert read.cos_ab == pytest.approx(float(a @ b / (a.norm() * b.norm())), abs=1e-5)
    # The null share: the bottom 1% of W_U diag(γ_f)'s right-singular directions.
    unembed = session.model.get_output_embeddings().weight.detach().float() * spec.final_norm.gain()
    vectors = torch.linalg.eigh((unembed.T @ unembed).double())[1].float()
    null = vectors[:, : max(1, int(0.01 * unembed.shape[1]))]
    assert read.null_share == pytest.approx(float((null.T @ w).pow(2).sum() / w.pow(2).sum()), rel=1e-3)
    # Logit-lens skew and kurtosis over the token mask; the − pole's skew flips.
    logits = _readout(session)(w[None]).float()[0][_keep(session)]
    skew, kurt = logit_moments(logits[None])
    assert read.poles[0].vocab_skew == pytest.approx(float(skew[0]), rel=1e-4)
    assert read.poles[1].vocab_skew == pytest.approx(-float(skew[0]), rel=1e-4)
    assert read.poles[0].vocab_kurt == read.poles[1].vocab_kurt == pytest.approx(float(kurt[0]), rel=1e-4)
    # Neutral firing: act_top per pole and the mean, from the unit statistics.
    stats = session.units.runtime.stats().layer(L)
    plus = read.poles[0].firing
    assert plus.neutral is not None
    assert plus.neutral["act_top"] == pytest.approx(float(stats.neuron_scale[0, n]), abs=1e-6)
    assert cast(dict[str, Any], read.poles[1].firing.neutral)["act_top"] == pytest.approx(float(stats.neuron_scale[1, n]), abs=1e-6)
    assert read.mean_act == pytest.approx(float(stats.neuron_mean[n]), abs=1e-6)
    # No catalog for the tiny model: the rate is absent with the reason.
    payload = read.to_dict()
    assert payload["poles"][0]["firing"]["rate"] is None
    assert "no neuron-lens catalog" in payload["poles"][0]["firing"]["rate_absent"]
    assert payload["catalog"]["bound"] is False and payload["poles"][0]["label"] is None


def test_last_layer_write_reads_the_logit_lens(lensed: Any) -> None:
    session, _r, _s = lensed
    last = len(session.layers) - 1
    read = cast(NeuronRead, session.units.read(f"neuron/L{last}n0", k=3))
    w = neuron_write(model_spec(session.model).blocks[last], 0)
    writes = read.poles[0].writes
    assert writes.r.present and writes.r.lens == "logit"
    probs = _r_probs(session, w[None], None)[0]
    assert _ids(writes.r) == probs.topk(3).indices.tolist()
    # mid has its R matrix: no logit read there.
    assert read.poles[0].fires_on.r.lens == "R:mid"


def test_hidden_hubs_rank_by_the_layer_contrast(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    L, n = 1, 3
    block = spec.blocks[L]
    read = cast(NeuronRead, session.units.read(f"neuron/L{L}n{n}", k=5, hide_hubs=True))
    keep = _keep(session)
    writes = neuron_writes(block).T          # [f, d]
    # R: the lens logits of every + write, z-scored per token across the layer.
    logits = _readout(session)(writes @ r.matrix(L, "out").T).float()
    mean, std = logits.mean(0), logits.std(0, correction=0)
    for i, sign in enumerate((1.0, -1.0)):
        z = ((sign * logits[n] - sign * mean) / std).masked_fill(~keep, -torch.inf)
        half = read.poles[i].writes.r
        assert half.unit == "layer_z" and read.poles[i].writes.hubs == "hidden"
        assert _ids(half) == z.topk(5).indices.tolist()
        assert [t.value for t in half.tokens] == pytest.approx(z.topk(5).values.tolist(), rel=1e-3)
    # S: unit covectors' raw scores over the + writes; − is minus the + z.
    u = writes @ s.matrix(L, "out")
    unit = u / u.norm(dim=-1, keepdim=True)
    scores = unit @ _centred(session).T
    mean, std = scores.mean(0), scores.std(0, correction=0)
    z = ((scores[n] - mean) / std).masked_fill(~keep, -torch.inf)
    assert _ids(read.poles[0].writes.s) == z.topk(5).indices.tolist()
    z_minus = ((-scores[n] + mean) / std).masked_fill(~keep, -torch.inf)
    assert _ids(read.poles[1].writes.s) == z_minus.topk(5).indices.tolist()
    # Fires on: R per pole's population, S over both poles together.
    plus, minus = neuron_reads(block, "+"), neuron_reads(block, "-")
    logits = _readout(session)(minus @ r.matrix(L, "mid").T).float()
    z = ((logits[n] - logits.mean(0)) / logits.std(0, correction=0)).masked_fill(~keep, -torch.inf)
    assert _ids(read.poles[1].fires_on.r) == z.topk(5).indices.tolist()
    both = torch.cat([plus, minus]) @ s.matrix(L, "mid")
    unit = both / both.norm(dim=-1, keepdim=True)
    scores = unit @ _centred(session).T
    z = ((scores[n] - scores.mean(0)) / scores.std(0, correction=0)).masked_fill(~keep, -torch.inf)
    assert _ids(read.poles[0].fires_on.s) == z.topk(5).indices.tolist()


def test_contrast_cache_persists_and_follows_the_lens(lensed: Any) -> None:
    session, _r, _s = lensed
    reader = session.units.reads
    first = session.units.read("neuron/L1n2", hide_hubs=True).to_dict()
    from drowse.units.reads import contrast_dir

    shards = sorted(p.relative_to(contrast_dir(session.model_id)) for p in contrast_dir(session.model_id).rglob("L*.safetensors"))
    assert len(shards) == 4  # R and S, writes and fires, layer 1
    reader.contrast.clear()
    assert session.units.read("neuron/L1n2", hide_hubs=True).to_dict() == first
    # A new S lens (another identity) builds new S shards; R's are reused.
    _install_lens(session, "S", seed=7, last_out=True)
    session.lens.select_source("local:s")
    session.units.read("neuron/L1n2", hide_hubs=True)
    after = list(contrast_dir(session.model_id).rglob("L*.safetensors"))
    assert len(after) == 6


def _count_some_tokens(session: Any, *, dropped: int) -> None:
    """Write corpus counts that keep every vocabulary id but the first ``dropped``."""
    from drowse.lens.token_counts import MIN_COUNT_TOKENS, tokenizer_fingerprint, write_corpus_counts

    rows = int(session.model.get_input_embeddings().weight.shape[0])
    counts = torch.full((rows,), 1_000, dtype=torch.int64)
    counts[:dropped] = 0
    write_corpus_counts(
        session.model_id, counts, corpus="test", documents=10,
        tokens=MIN_COUNT_TOKENS, tokenizer=tokenizer_fingerprint(session.tokenizer),
    )


def test_unit_reads_follow_a_token_mask_change(lensed: Any) -> None:
    # Regression: the in-memory contrast slot answered before its record (which
    # carries the mask digest) was compared, so a mask change mid-session kept
    # the old population stats: neuron S z-scores went stale and a head read
    # with hidden hubs subtracted an old-mask mean from a new-mask z (a size
    # mismatch, a 500).
    session, _r, _s = lensed
    session.units.read("neuron/L1n2", hide_hubs=True)
    session.units.read("head/L1h2", k=3, hide_hubs=True)
    _count_some_tokens(session, dropped=9)
    neuron = session.units.read("neuron/L1n2", hide_hubs=True).to_dict()
    head = session.units.read("head/L1h2", k=3, hide_hubs=True).to_dict()
    session.units.reads.contrast.clear()
    assert session.units.read("neuron/L1n2", hide_hubs=True).to_dict() == neuron
    assert session.units.read("head/L1h2", k=3, hide_hubs=True).to_dict() == head


def test_token_logits_match_the_readout(lensed: Any) -> None:
    session, _r, _s = lensed
    rows = torch.randn(5, 32, generator=torch.Generator().manual_seed(3))
    ids = torch.tensor([3, 40, 77])
    full = _readout(session)(rows).float()
    assert token_logits(_readout(session), rows, ids) == pytest.approx(full[:, ids], rel=1e-5, abs=1e-5)
    assert full_logits(_readout(session), rows) == pytest.approx(full, rel=1e-5, abs=1e-5)


def test_low_precision_logits_agree_between_one_token_and_all(lensed: Any) -> None:
    """Regression: on a bf16 model the population ``logsumexp`` came from the
    readout's bf16 logits and a pole's token logit from an fp32 product, so
    ``top`` reported probabilities above 1; both are now the fp32 product."""
    import copy

    from drowse.model.readout import LogitReadout

    session, _r, _s = lensed
    readout = LogitReadout(copy.deepcopy(session.model).to(torch.bfloat16))
    rows = 4 * torch.randn(64, 32, generator=torch.Generator().manual_seed(5))
    ids = torch.arange(0, 200, 7)
    full = full_logits(readout, rows)
    assert token_logits(readout, rows, ids) == pytest.approx(full[:, ids], rel=1e-6, abs=1e-6)
    probs = (token_logits(readout, rows, ids) - full.logsumexp(-1, keepdim=True)).exp()
    assert float(probs.max()) <= 1.0


# -- heads ---------------------------------------------------------------------


def _ov_by_hand(session: Any, X: torch.Tensor, A_rows: torch.Tensor, block: Any, h: int) -> torch.Tensor:
    """The whole token map ``M = A Bᵀ`` and each token's own-column z."""
    A = A_rows @ head_write_matrix(block, h)
    B = X @ head_read_rows(block, h).value.T
    M = A @ B.T                                   # [u, v]
    return (M.diagonal() - M.mean(0)) / M.std(0, correction=0)


def test_head_ov_copies_and_copy_score_by_hand(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    L, h = 2, 1
    block = spec.blocks[L]
    read = cast(HeadRead, session.units.read(f"head/L{L}h{h}", k=4))
    assert read.kv_head == block.attn.kv_head(h)
    keep = _keep(session)
    kept = keep.nonzero().squeeze(1)
    E = _centred(session)[kept]
    X = E @ s.matrix(L - 1, "out").T
    gain = spec.final_norm.gain()
    unembed = session.model.get_output_embeddings().weight.detach().float()[kept] * gain
    z_r = _ov_by_hand(session, X, unembed @ r.matrix(L, "mid"), block, h)
    assert [kept[i].item() for i in z_r.topk(4).indices] == _ids(read.copies.r)
    assert [t.value for t in read.copies.r.tokens] == pytest.approx(z_r.topk(4).values.tolist(), rel=1e-3)
    assert [kept[i].item() for i in (-z_r).topk(4).indices] == _ids(read.suppresses.r)
    assert read.suppresses.r.tokens[0].value == pytest.approx(float(z_r.min()), rel=1e-3)
    z_s = _ov_by_hand(session, X, E @ s.matrix(L, "mid").T, block, h)
    assert [kept[i].item() for i in z_s.topk(4).indices] == _ids(read.copies.s)
    assert (read.copies.r.lens, read.copies.s.lens) == ("R:mid", "S:mid")
    # The copy score: Σλ/Σ|λ| over the map's nonzero eigenvalues.
    A = (unembed @ r.matrix(L, "mid")) @ head_write_matrix(block, h)
    B = X @ head_read_rows(block, h).value.T
    lam = torch.linalg.eigvals((B.T @ A).double())
    assert read.copy_score["R"] == pytest.approx(float(lam.real.sum() / lam.abs().sum()), rel=1e-4)
    assert own_token_z(A, B) == pytest.approx(z_r, rel=1e-3, abs=1e-4)
    # The read sums Bᵀ A chunk by chunk: the same score up to fp32 order.
    assert copy_score(A, B) == pytest.approx(read.copy_score["R"], rel=1e-4)


def _energy_z(rows: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Vocabulary z of the share ``‖P x̂_v‖`` of each row's direction ``P`` keeps."""
    energy = (rows @ P.T).norm(dim=-1) / rows.norm(dim=-1)
    return (energy - energy.mean()) / energy.std(correction=0)


def test_head_inputs_by_hand(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    L, h = 2, 3
    block = spec.blocks[L]
    read = cast(HeadRead, session.units.read(f"head/L{L}h{h}", k=4))
    keep = _keep(session)
    kept = keep.nonzero().squeeze(1)
    X = _centred(session)[kept] @ s.matrix(L - 1, "out").T
    # The R input of token v: the direction that says v at L − 1's output,
    # J^{out⊤}_{L−1}(γ_f ⊙ W_U[v]), as a row (γ_f ⊙ W_U[v]) J.
    J = r.matrix(L - 1, "out")
    unembed = session.model.get_output_embeddings().weight.detach().float()[kept] * spec.final_norm.gain()
    Y = unembed @ J
    for role in ("query", "key", "value"):
        place = read.inputs[role]
        assert (place.layer, place.site) == (L - 1, "out")
        P = getattr(head_read_rows(block, h), role)
        for half, rows, lens in ((place.s, X, "S:out"), (place.r, Y, "R:out")):
            z = _energy_z(rows, P)
            assert half.unit == "energy_z" and half.lens == lens
            assert _ids(half) == [kept[i].item() for i in z.topk(4).indices]
            assert [t.value for t in half.tokens] == pytest.approx(z.topk(4).values.tolist(), rel=1e-4)
        # J's orientation matters: the transposed read ranks other values.
        transposed = _energy_z(unembed @ J.T, P).topk(4).values
        assert not torch.allclose(transposed, _energy_z(Y, P).topk(4).values, rtol=1e-3)
    # A head at L0 reads the embedding: S over ẽ, no R read.
    first = cast(HeadRead, session.units.read("head/L0h0", k=2))
    assert first.inputs["key"].s.lens == "embedding" and not first.inputs["key"].r.present
    z = _energy_z(_centred(session)[kept], head_read_rows(spec.blocks[0], 0).key)
    assert _ids(first.inputs["key"].s) == [kept[i].item() for i in z.topk(2).indices]


def test_a_head_read_streams_the_vocabulary_through_small_factors(
    lensed: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: one head read held several ``[V, d]`` fp32 tensors at once
    (the footprints, both OV A sides, the R inputs; about 15 GB on Qwen3-8B).
    Every product now takes the ``[d, hd]`` factors first and streams the
    vocabulary through them in chunks: no tensor the read makes, its
    hidden-hub population included, spans the vocabulary at residual width."""
    import shutil

    from torch.overrides import TorchFunctionMode

    import drowse.units.reads as reads_module
    from drowse.units.reads import contrast_dir

    session, _r, _s = lensed
    expected = cast(HeadRead, session.units.read("head/L2h1", k=4, hide_hubs=True)).to_dict()
    session.units.reads.contrast.clear()
    shutil.rmtree(contrast_dir(session.model_id))
    monkeypatch.setattr(reads_module, "VOCAB_CHUNK", 16)
    n_vocab = int(_keep(session).sum())
    width = int(model_spec(session.model).d_model)
    shapes: list[tuple[int, ...]] = []

    class Shapes(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None,
        ) -> Any:
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor):
                shapes.append(tuple(out.shape))
            return out

    with Shapes():
        got = cast(HeadRead, session.units.read("head/L2h1", k=4, hide_hubs=True)).to_dict()
    wide = [
        shape for shape in shapes
        if shape and max(shape) >= n_vocab and math.prod(shape) >= n_vocab * width // 2
    ]
    assert shapes and not wide
    # Chunked or not, the same read (fp32 summation order aside).
    for place in ("copies", "suppresses"):
        for family in ("s", "r"):
            assert [t["token_id"] for t in got[place][family]["tokens"]] == [
                t["token_id"] for t in expected[place][family]["tokens"]
            ]
            assert [t["value"] for t in got[place][family]["tokens"]] == pytest.approx(
                [t["value"] for t in expected[place][family]["tokens"]], rel=1e-4, abs=1e-5,
            )
    assert got["copy_score"] == pytest.approx(expected["copy_score"], rel=1e-4)


def test_head_hidden_hubs_contrast_across_heads(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    L, h = 1, 2
    block = spec.blocks[L]
    read = cast(HeadRead, session.units.read(f"head/L{L}h{h}", k=3, hide_hubs=True))
    kept = _keep(session).nonzero().squeeze(1)
    E = _centred(session)[kept]
    X = E @ s.matrix(L - 1, "out").T
    side = E @ s.matrix(L, "mid").T
    per_head = torch.stack([_ov_by_hand(session, X, side, block, i) for i in range(block.n_heads)])
    zc = (per_head[h] - per_head.mean(0)) / per_head.std(0)
    assert read.copies.s.unit == "layer_z"
    assert _ids(read.copies.s) == [kept[i].item() for i in zc.topk(3).indices]
    # Regression: inputs never contrast across heads (a KV group's heads share
    # key and value rows, so every token saturated at one z); they stay raw.
    raw = cast(HeadRead, session.units.read(f"head/L{L}h{h}", k=3))
    for role in ("query", "key", "value"):
        assert read.inputs[role].hubs == "shown" and read.inputs[role].s.unit == "energy_z"
        assert read.inputs[role].to_dict() == raw.inputs[role].to_dict()


# -- top and find ----------------------------------------------------------------


def test_top_ranks_poles_by_the_weighted_layer_z(lensed: Any) -> None:
    session, r, s = lensed
    spec = model_spec(session.model)
    token = "e"
    from drowse.lens.jlens import resolve_word_token

    v = resolve_word_token(session.tokenizer, token)
    L = 1
    result = session.units.top(token, by="writes", layers=[L], k=5)
    assert result.token_id == v and result.by == "writes"
    block = spec.blocks[L]
    writes = neuron_writes(block).T
    logits = _readout(session)(writes @ r.matrix(L, "out").T).float()
    z_r = (logits[:, v] - logits[:, v].mean()) / logits.std(0, correction=0)[v]
    u = writes @ s.matrix(L, "out")
    unit = u / u.norm(dim=-1, keepdim=True)
    scores = unit @ _centred(session).T
    z_s = (scores[:, v] - scores[:, v].mean()) / scores.std(0, correction=0)[v]
    w_s, w_r = kind_weights(len(session.layers), session.lens.band)[L]
    combined = torch.cat([w_r * z_r + w_s * z_s, -(w_r * z_r + w_s * z_s)])
    best = combined.topk(5)
    f = writes.shape[0]
    expected = [(int(i) % f, "+" if int(i) < f else "-") for i in best.indices]
    assert [(hit.ref.index, hit.pole) for hit in result.hits] == expected
    assert [hit.score for hit in result.hits] == pytest.approx(best.values.tolist(), rel=1e-4)
    assert all(0.0 <= hit.r["probability"] <= 1.0 for hit in result.hits)
    top = result.hits[0]
    probs = _r_probs(session, (writes[top.ref.index] * (1 if top.pole == "+" else -1))[None], r.matrix(L, "out"))[0]
    assert top.r["probability"] == pytest.approx(float(probs[v]), rel=1e-3)
    assert result.layers[L]["weights"] == {"S": pytest.approx(w_s, abs=1e-6), "R": pytest.approx(w_r, abs=1e-6)}


def test_top_by_fires_reads_the_read_directions(lensed: Any) -> None:
    session, _r, _s = lensed
    result = session.units.top("a", by="fires", layers="all", k=4)
    assert len(result.hits) == 4
    assert {hit.ref.layer for hit in result.hits} <= set(range(len(session.layers)))
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    from drowse.errors import RequestError

    with pytest.raises(RequestError, match="by must be"):
        session.units.top("a", by="says")
    # Regression: a string spec goes through the one layer parser (ranges too).
    ranged = session.units.top("a", layers="1-2", k=2)
    assert set(ranged.layers) == {1, 2}
    with pytest.raises(RequestError, match="bad layers"):
        session.units.top("a", layers="1-")


def test_find_recovers_a_neuron_write(lensed: Any) -> None:
    session, _r, _s = lensed
    L, n = 2, 11
    w = neuron_write(model_spec(session.model).blocks[L], n)
    found = session.units.find(w, layer=L, k=3).to_dict()
    row = found["layers"][str(L)]
    assert row["share"] == pytest.approx(1.0, abs=1e-4)
    assert row["neurons"][0]["unit"] == f"neuron/L{L}n{n}" and row["neurons"][0]["pole"] == "+"
    negative = session.units.find(-w, layer=L, k=1).to_dict()["layers"][str(L)]["neurons"][0]
    assert negative["pole"] == "-" and negative["coeff"] < 0
    # A selector goes through the lens decomposition's neurons dictionary.
    session.lens.decompose("jlens/a", dictionary="tokens-out")  # the R lens resolves jlens/
    by_selector = session.units.find("jlens/a", k=2, layers=[1])
    assert set(by_selector.layers) == {1} and len(by_selector.layers[1]["neurons"]) <= 2


def test_population_artifacts_reproduce_top_without_the_vocabulary(lensed: Any) -> None:
    """The per-layer artifacts a client ships offline: the population
    scalars, contrast statistics and whitened write norms, by hand.  With
    them one token's ``top`` z needs that token's logit alone."""
    from drowse.lens.decompose import NeuronWrites

    session, r, s = lensed
    spec = model_spec(session.model)
    L = 1
    block = spec.blocks[L]
    reads = session.units.reads
    artifacts = reads.population_artifacts(L)
    writes = neuron_writes(block).T
    a, b = neuron_read_rows(block)
    f = writes.shape[0]
    # R: the final norm's scale of each transported write, its lse and the
    # per-token mean and std across the layer's writes.
    x = writes @ r.matrix(L, "out").T
    rho = torch.rsqrt(x.pow(2).mean(-1) + spec.final_norm.eps)
    torch.testing.assert_close(artifacts["R.writes.rho"], rho, rtol=1e-5, atol=1e-6)
    logits = full_logits(_readout(session), x)
    torch.testing.assert_close(artifacts["R.writes.lse"], logits.logsumexp(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(artifacts["R.writes.lse_neg"], (-logits).logsumexp(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(artifacts["R.writes.mean"], logits.mean(0), rtol=1e-5, atol=1e-5)
    assert set(k for k in artifacts if k.startswith("R.fires+")) == {
        "R.fires+.rho", "R.fires+.lse", "R.fires+.mean", "R.fires+.std",
    }
    # S: ‖u‖ and the footprint std per row; fires pooled over both poles.
    emb = session.lens.token_embeddings()
    u = writes @ s.matrix(L, "out")
    torch.testing.assert_close(artifacts["S.writes.norm"], u.norm(dim=-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(artifacts["S.writes.footprint_std"], emb.variance(u).sqrt(), rtol=1e-4, atol=1e-6)
    fires = torch.cat([a + b, a - b]) @ s.matrix(L, "mid")
    unit = fires / fires.norm(dim=-1, keepdim=True)
    pooled = unit @ emb.centred(range(emb.vocab)).T
    torch.testing.assert_close(artifacts["S.fires.mean"], pooled.mean(0), rtol=1e-4, atol=1e-5)
    assert "S.fires+.mean" not in artifacts and artifacts["S.fires+.norm"].shape == (f,)
    # The top shortcut: token v's z from its own logits and the artifacts.
    v = int(session.units.top("e", by="writes", layers=[L], k=1).token_id)
    lg = token_logits(_readout(session), x, torch.tensor([v]))[:, 0]
    z = (lg - artifacts["R.writes.mean"][v]) / artifacts["R.writes.std"][v]
    z_full = (logits[:, v] - logits[:, v].mean()) / logits[:, v].std(correction=0)
    torch.testing.assert_close(z, z_full, rtol=1e-4, atol=1e-4)
    # Heads: the OV contrast per family over the head-read vocabulary.
    kept = reads.head_kept_ids()
    assert artifacts["R.heads.ov.mean"].shape == artifacts["S.heads.ov.std"].shape == kept.shape
    assert not any(key.startswith("heads.norm_derivative") for key in artifacts)  # llama: no head norms
    # find's whitened write norms.
    expected = NeuronWrites(neuron_writes(block), session.whitener, L).norms()
    torch.testing.assert_close(artifacts["neurons.write_norms"], expected.cpu())
    # The null basis is public too: orthonormal columns, the bottom 1%.
    null = reads.null_basis()
    torch.testing.assert_close(null.T @ null, torch.eye(null.shape[1]), atol=1e-5, rtol=0)


def test_population_artifacts_leave_out_absent_halves(make_tiny_session: Any) -> None:
    session = make_tiny_session()
    _install_lens(session, "R", seed=1, last_out=False)
    session.lens.select_source("local:r")
    artifacts = session.units.reads.population_artifacts(1)
    # Without an S lens there is no S half, and no head contrast past L0
    # (a head's input side reads through S at L − 1 out).
    assert not any(key.startswith("S.") or ".heads." in key for key in artifacts)
    assert {"R.writes.rho", "R.fires-.lse", "neurons.write_norms"} <= set(artifacts)
    assert "R.heads.ov.mean" in session.units.reads.population_artifacts(0)  # L0 reads the embedding
    from drowse.errors import RequestError

    with pytest.raises(RequestError, match="outside"):
        session.units.reads.population_artifacts(len(session.layers))


def test_refusals(lensed: Any) -> None:
    session, _r, _s = lensed
    from drowse.errors import RequestError
    from drowse.model.units import UnitIndexError

    with pytest.raises(UnitIndexError, match="outside"):
        session.units.read("neuron/L1n9999")
    with pytest.raises(RequestError, match="neuron/L"):
        session.units.read("jlens/a")
    with pytest.raises(RequestError, match="k must be"):
        session.units.read("neuron/L1n1", k=0)
    with pytest.raises(RequestError, match="layer"):
        session.units.find(torch.ones(32))


def test_reads_without_lenses_are_absent_halves(make_tiny_session: Any) -> None:
    session = make_tiny_session()
    read = cast(NeuronRead, session.units.read("neuron/L1n0"))
    writes: PlaceRead = read.poles[0].writes
    assert not writes.r.present and not writes.s.present
    head = cast(HeadRead, session.units.read("head/L1h0"))
    assert not head.copies.r.present and "S lens" in str(head.copies.r.absent)
    payload = head.to_dict()
    assert payload["kind"] == "head" and payload["inputs"]["query"]["s"]["absent"]
    assert HeadRef(layer=1, index=0).qualified == payload["unit"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_copy_score_takes_device_factors_on_mps() -> None:
    """Regression: the eigenvalues were cast to float64 on the device, which
    MPS lacks; the factors go to the host first."""
    gen = torch.Generator().manual_seed(0)
    A, B = torch.randn(50, 4, generator=gen), torch.randn(50, 4, generator=gen)
    assert copy_score(A.to("mps"), B.to("mps")) == pytest.approx(copy_score(A, B), rel=1e-5)


# -- Gemma 4: head norms ---------------------------------------------------------


def _mean_norm_derivative(block: Any, role: str, rows: torch.Tensor) -> torch.Tensor:
    """``[heads, hd, hd]``: by autograd through the model's own head norm, its
    Jacobian at each residual's projection, weighted by the input norm's
    scale ``1/ρ`` and averaged over ``rows`` (neuron-lens's mean true
    derivative)."""
    attn = block.attn
    norm = {"query": attn.q_norm, "key": attn.k_norm, "value": attn.v_norm}[role]
    proj = {"query": attn.q, "key": attn.k, "value": attn.k if attn.v is None else attn.v}[role]
    hd = attn.head_dim
    gain_in = block.attn_in_norm.gain()
    total: torch.Tensor | None = None
    for row in rows:
        inv_rho = torch.rsqrt(row.pow(2).mean() + block.attn_in_norm.eps)
        x = proj(row * inv_rho * gain_in).detach().view(-1, hd)
        jac = torch.stack([
            torch.autograd.functional.jacobian(lambda y: norm.module(y[None])[0].float(), x[g])
            for g in range(x.shape[0])
        ])
        total = inv_rho * jac if total is None else total + inv_rho * jac
    assert total is not None
    return total / rows.shape[0]


def test_gemma4_head_reads_go_through_the_head_norms(tmp_path: Path) -> None:
    """Gemma 4's value norm is a true derivative averaged over the neutral
    residuals entering the block (as neuron-lens's OV reads apply it), so
    the copies, suppressions and copy score line up with the catalog's
    ``ov_R``/``ov_S``; the query and key inputs pass their q/k norms the
    same way."""
    from tests.support.sessions import TinyCheckpoint, TinySessionFactory
    from tests.support.tiny_models import write_tiny_checkpoint
    from tests.support.units import GEMMA4_MIXED_HEADS

    checkpoint = TinyCheckpoint(
        write_tiny_checkpoint(tmp_path / "tiny-g4", GEMMA4_MIXED_HEADS), GEMMA4_MIXED_HEADS,
    )
    factory = TinySessionFactory(checkpoint, tmp_path / "seed")
    try:
        session = factory()
        r: JacobianLens = _install_lens(session, "R", seed=1, last_out=False)
        s: JacobianLens = _install_lens(session, "S", seed=2, last_out=True)
        session.lens.select_source("local:r")
        session.lens.select_source("local:s")
        spec = model_spec(session.model)
        L, h = 1, 2
        block = spec.blocks[L]
        attn = block.attn
        assert attn.v_norm is not None and not attn.v_norm.weighted and attn.q_norm is not None
        read = cast(HeadRead, session.units.read(f"head/L{L}h{h}", k=4))

        whitener = session.whitener
        centred, _small, _ridge = whitener.woodbury_factors(L - 1, device=torch.device("cpu"), dtype=torch.float32)
        rows = centred + whitener.layer_means[L - 1]
        D_v = _mean_norm_derivative(block, "value", rows)[attn.kv_head(h)]
        D_q = _mean_norm_derivative(block, "query", rows)[h]
        # The derivative is not a multiple of the identity: it turns B.
        assert float((D_v - D_v.diagonal().mean() * torch.eye(D_v.shape[0])).abs().max()) > 1e-3
        # The offline artifacts carry the same derivatives a head read applies.
        artifacts = session.units.reads.population_artifacts(L)
        torch.testing.assert_close(
            artifacts["heads.norm_derivative.value"][attn.kv_head(h)], D_v, atol=1e-5, rtol=1e-4,
        )
        torch.testing.assert_close(artifacts["heads.norm_derivative.query"][h], D_q, atol=1e-5, rtol=1e-4)

        emb = session.lens.token_embeddings()
        kept = emb.keep.nonzero().squeeze(1)
        E = emb.centred(kept)
        X = E @ s.matrix(L - 1, "out").T
        value = head_read_rows(block, h).value
        B = X @ value.T @ D_v.T
        write = head_write_matrix(block, h)
        unembed = session.model.get_output_embeddings().weight.detach().float()[kept] * spec.final_norm.gain()
        for half, A in (
            (read.copies.r, unembed @ r.matrix(L, "mid") @ write),
            (read.copies.s, E @ s.matrix(L, "mid").T @ write),
        ):
            z = own_token_z(A, B)
            assert _ids(half) == [kept[i].item() for i in z.topk(4).indices]
            assert [t.value for t in half.tokens] == pytest.approx(z.topk(4).values.tolist(), rel=1e-3)
        A_r = unembed @ r.matrix(L, "mid") @ write
        assert read.copy_score["R"] == pytest.approx(copy_score(A_r, B), rel=1e-4)
        # Without the value norm's derivative the score would differ.
        assert copy_score(A_r, X @ value.T) != pytest.approx(read.copy_score["R"], rel=1e-3)
        # Inputs: each projection through its head norm.
        for role, P in (("value", D_v @ value), ("query", D_q @ head_read_rows(block, h).query)):
            z = _energy_z(X, P)
            assert _ids(read.inputs[role].s) == [kept[i].item() for i in z.topk(4).indices]

        # At L0 the norms average over the kept tokens' embeddings.
        first = cast(HeadRead, session.units.read("head/L0h1", k=3))
        block0 = spec.blocks[0]
        embedded = session.model.get_input_embeddings().weight.detach().float()[kept] * emb.scale
        D0 = _mean_norm_derivative(block0, "value", embedded)[block0.attn.kv_head(1)]
        z = _energy_z(E, D0 @ head_read_rows(block0, 1).value)
        assert _ids(first.inputs["value"].s) == [kept[i].item() for i in z.topk(3).indices]
    finally:
        factory.close()


def test_head_norms_need_the_neutral_residuals(tmp_path: Path) -> None:
    """A block with head norms (Qwen3's q/k norms) reads its heads through
    the neutral residuals entering it; a reader without them refuses rather
    than read the projections unnormed."""
    from collections.abc import Iterator
    from contextlib import contextmanager

    from drowse.errors import WhitenerError
    from drowse.units.reads import UnitReader
    from tests.support.sessions import TinyCheckpoint, TinySessionFactory

    factory = TinySessionFactory(TinyCheckpoint.write(tmp_path, "qwen3"), tmp_path / "seed")
    try:
        session = factory()
        reads = session.units.reads

        @contextmanager
        def exclusive(_message: str, **_kwargs: Any) -> Iterator[None]:
            yield

        bare = UnitReader(
            reads.runtime, reads.lens, model_id=session.model_id, n_layers=len(session.layers),
            exclusive=exclusive, catalog=reads.catalog, weights=lambda: "weights",
        )
        with pytest.raises(WhitenerError, match="neutral residuals at L0"):
            bare.read("head/L1h0")
        assert session.units.read("head/L1h0").to_dict()["kind"] == "head"
    finally:
        factory.close()
