"""Unit statistics on the neutral pass (neutral cache v5).

The streaming accumulator is checked against a hand computation (each row
alone, unpadded, every activation kept), on llama (plain blocks) and Gemma 3
(sandwich norms: the head write norm carries the post-attention norm's gain
and per-position scale).  The cache rides the statistics in its shards, and a
model the block description refuses still gets a complete cache with units
marked unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.errors import UnsupportedBlockError
from drowse.model.blocks import model_spec
from drowse.model.units import head_write_matrix
from drowse.units.stats import (
    HEAD_SCALE_FROM,
    UNIT_POSITIONS_FROM,
    UnitStatsAccumulator,
    top_median,
)
from tests.support.tiny_models import tiny_model

pytestmark = pytest.mark.usefixtures("isolated_home")


def _tiny_model(_tmp_path: Path, family: str) -> Any:
    return tiny_model(family)


def _rows(vocab: int, lengths: list[int], seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(3, vocab, (n,), generator=gen) for n in lengths]


def _hand_stats(model: Any, rows: list[torch.Tensor], k: int) -> dict[int, dict[str, torch.Tensor]]:
    """Each row alone: every down-proj input and o-proj input/output kept."""
    spec = model_spec(model)
    acts: dict[int, list[torch.Tensor]] = {b.layer: [] for b in spec.blocks}
    zs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {b.layer: [] for b in spec.blocks}
    def keep_act(layer: int) -> Any:
        def hook(_m: torch.nn.Module, args: tuple[Any, ...]) -> None:
            acts[layer].append(args[0][0].detach().clone())
        return hook

    def keep_z(layer: int) -> Any:
        def hook(_m: torch.nn.Module, args: tuple[Any, ...], out: torch.Tensor) -> None:
            zs[layer].append((args[0][0].detach().clone(), out[0].detach().clone()))
        return hook

    handles = []
    for block in spec.blocks:
        handles.append(block.mlp.down.register_forward_pre_hook(keep_act(block.layer)))
        handles.append(block.attn.o.register_forward_hook(keep_z(block.layer)))
    try:
        with torch.inference_mode():
            for ids in rows:
                model(input_ids=ids[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    out: dict[int, dict[str, torch.Tensor]] = {}
    for block in spec.blocks:
        L = block.layer
        n_heads, hd = block.attn.n_heads, block.attn.head_dim
        kept = [a[UNIT_POSITIONS_FROM:] for a in acts[L]]
        flat = torch.cat(kept)
        peaks_pos = torch.stack([a.max(0).values for a in kept if len(a)])
        peaks_neg = torch.stack([(-a).max(0).values for a in kept if len(a)])
        top_pos = peaks_pos.sort(0, descending=True).values[:k]
        top_neg = peaks_neg.sort(0, descending=True).values[:k]
        z_flat = torch.cat([z[UNIT_POSITIONS_FROM:] for z, _ in zs[L]]).view(-1, n_heads, hd)
        writes = torch.stack([head_write_matrix(block, h) for h in range(n_heads)])
        head_peaks = []
        for z, o in zs[L]:
            zh = z.view(-1, n_heads, hd)
            norm = torch.einsum("dhi,thi->thd", writes.permute(1, 0, 2), zh).norm(dim=-1)
            if block.attn_out_norm is not None:
                r = torch.rsqrt(o.pow(2).mean(-1) + block.attn_out_norm.eps)
                norm = norm * r[:, None]
            if len(z) > HEAD_SCALE_FROM:
                head_peaks.append(norm[HEAD_SCALE_FROM:].max(0).values)
        head_top = torch.stack(head_peaks).sort(0, descending=True).values[:k]
        out[L] = {
            "neuron_mean": flat.mean(0),
            "neuron_scale": torch.stack([
                top_median(top_pos, len(top_pos)),
                top_median(top_neg, len(top_neg)),
            ]),
            "head_mean": z_flat.mean(0),
            "head_scale": top_median(head_top, len(head_top)),
        }
    return out


@pytest.mark.parametrize("family", ["llama", "gemma3_text"])
def test_streaming_stats_match_a_hand_computation(tmp_path: Path, family: str) -> None:
    model = _tiny_model(tmp_path, family)
    vocab = int(model.config.vocab_size)
    lengths = [30, 19, 26, 12, 33, 21, 17]
    rows = _rows(vocab, lengths)
    k = 3
    spec = model_spec(model)
    acc = UnitStatsAccumulator(spec, device=torch.device("cpu"), k=k)
    with acc:
        # Two right-padded chunks, one with a discarded (retried) attempt.
        for chunk in (rows[:4], rows[4:]):
            lens = [len(r) for r in chunk]
            ids = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True, padding_value=0)
            mask = (torch.arange(ids.shape[1])[None] < torch.tensor(lens)[:, None]).long()
            acc.begin_chunk(lens)
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            if chunk is rows[4:]:
                acc.discard_chunk()
                acc.begin_chunk(lens)
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
            acc.commit_chunk()
    stats = acc.finish()
    hand = _hand_stats(model, rows, k)
    assert stats.n_documents == len(rows)
    for layer, expected in hand.items():
        got = stats.layer(layer).tensors()
        for name, tensor in expected.items():
            torch.testing.assert_close(got[name], tensor, rtol=1e-5, atol=1e-5, msg=name)


def test_top_median_takes_the_documents_seen() -> None:
    top = torch.tensor([[5.0], [3.0], [1.0], [float("-inf")]])
    assert top_median(top, 3).item() == 3.0
    assert top_median(top, 2).item() == 3.0  # the lower median of (5, 3)
    assert top_median(top, 0).item() == 0.0
    assert top_median(torch.tensor([[-2.0]]), 1).item() == 0.0  # clamped


def test_a_failing_hook_marks_units_failed_without_raising(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path, "llama")
    acc = UnitStatsAccumulator(model_spec(model), device=torch.device("cpu"), k=2)
    with acc:
        acc.begin_chunk([5, 7])  # but the forward below has another shape
        with torch.inference_mode():
            model(input_ids=torch.ones(1, 3, dtype=torch.long), use_cache=False)
        acc.commit_chunk()
    assert acc.failure is not None
    with pytest.raises(RuntimeError):
        acc.finish()


# ------------------------------------------------------------ the cache


def test_the_session_neutral_cache_carries_unit_stats(tiny_session: Any) -> None:
    from drowse.base.paths import model_dir

    stats = tiny_session.units.runtime.stats()
    spec = tiny_session.units.runtime.spec()
    assert sorted(stats.layers) == list(range(spec.n_layers))
    block = spec.blocks[0]
    layer = stats.layer(0)
    assert layer.neuron_mean.shape == (block.d_ff,)
    assert layer.neuron_scale.shape == (2, block.d_ff)
    assert layer.head_mean.shape == (block.n_heads, block.attn.head_dim)
    assert layer.head_scale.shape == (block.n_heads,)
    assert bool((layer.neuron_scale > 0).all()) and bool((layer.head_scale > 0).all())
    sidecar = json.loads(
        (model_dir(tiny_session.model_id) / "neutral_activations.json").read_text()
    )
    assert sidecar["format_version"] == 5
    assert sidecar["units"]["available"] is True
    assert sidecar["units"]["n_documents"] == stats.n_documents > 0


def test_an_undescribed_model_still_gets_its_cache_with_units_unavailable(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.model.blocks as blocks_mod
    from drowse.fit.neutral import load_or_compute_neutral_cache
    from drowse.store.neutral import load_neutral_cache

    def refuse(_model: Any) -> Any:
        raise UnsupportedBlockError("mixture-of-experts block ('experts')", layer=0)

    monkeypatch.setattr(blocks_mod, "model_spec", refuse)
    before = load_neutral_cache(tiny_session.model_id).activations
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id, force=True,
    )
    assert cache.units.tensors is None
    assert cache.units.reason is not None and "experts" in cache.units.reason
    assert cache.sidecar["units"] == {"available": False, "reason": cache.units.reason}
    reloaded = load_neutral_cache(tiny_session.model_id)
    assert reloaded.units.tensors is None
    for layer, rows in before.items():
        torch.testing.assert_close(reloaded.activations[layer], rows)


def test_an_older_cache_is_recomputed_not_migrated(tiny_session: Any) -> None:
    from drowse.base.paths import model_dir
    from drowse.fit.neutral import load_or_compute_neutral_cache
    from drowse.store.neutral import load_neutral_cache

    pointer = model_dir(tiny_session.model_id) / "neutral_activations.json"
    sidecar = json.loads(pointer.read_text())
    sidecar["format_version"] = 4
    del sidecar["units"]
    pointer.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        load_neutral_cache(tiny_session.model_id)
    cache = load_or_compute_neutral_cache(
        tiny_session.model, tiny_session.tokenizer, tiny_session.layers,
        model_id=tiny_session.model_id,
    )
    assert cache.sidecar["format_version"] == 5
    assert cache.units.tensors is not None


def test_a_malformed_units_block_is_rejected(tiny_session: Any) -> None:
    from drowse.base.paths import model_dir
    from drowse.store.neutral import load_neutral_cache

    pointer = model_dir(tiny_session.model_id) / "neutral_activations.json"
    sidecar = json.loads(pointer.read_text())
    sidecar["units"]["schema"]["0"]["neuron_mean"] = [999]
    pointer.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="unit"):
        load_neutral_cache(tiny_session.model_id)
