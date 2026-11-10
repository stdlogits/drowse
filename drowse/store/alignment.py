"""Per-(source, target) cross-model alignment-map cache.

A fitted :class:`~drowse.geometry.procrustes.LayerAlignment` map persists under
the *target* model's directory as ``alignments/<safe_src>.json`` (an atomic
pointer) plus one immutable ``left``/``right``/``offset`` safetensors shard per
layer (:mod:`drowse.base.shards`).  The pointer embeds both models' neutral
cache identities (:func:`drowse.store.neutral.neutral_cache_identity`), so a
re-captured neutral cache on either side invalidates the map.  The reader
needs the keys it uses and ignores any other (6.1's per-shard
``tensor_sha256`` included).  Readers never mutate the store; superseded
generations are collected by the next writer.
"""
from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from safetensors.torch import (
    load as load_safetensors,
    save as save_safetensors,
)

from drowse.base.atomic import fsync_directory, write_bytes_atomic, write_json_atomic
from drowse.base.paths import model_dir, safe_model_id
from drowse.base.shards import (
    cleanup_generations,
    fit_lock,
    generation_path,
    json_pointer_matches as _json_pointer_matches,
    shard_paths,
)
from drowse.geometry.procrustes import AlignmentMap, LayerAlignment

log = logging.getLogger(__name__)

_ALIGNMENT_LABEL = "alignment cache"
_ALIGNMENT_CACHE_FORMAT_VERSION = 5
_ALIGNMENT_METHOD = "sharded_factorized_affine_alignment"


def _alignment_anchor_paths(
    src_model_id: str, tgt_model_id: str,
) -> tuple[Path, Path]:
    """Stable lock anchor and atomic v5 pointer sidecar."""
    md = model_dir(tgt_model_id)
    al_dir = md / "alignments"
    src_safe = safe_model_id(src_model_id)
    return (al_dir / f"{src_safe}.safetensors", al_dir / f"{src_safe}.json")


def _alignment_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    return shard_paths(anchor, sidecar, layers, label=_ALIGNMENT_LABEL)


def _cleanup_alignment_generations(anchor: Path, sidecar: Mapping[str, Any]) -> None:
    cleanup_generations(anchor, sidecar, label=_ALIGNMENT_LABEL)


def _identity_matches(stored: Any, current: Mapping[str, Any]) -> bool:
    """Whether a stored neutral identity carries every current key's value.

    Keys the current identity no longer names (6.1's ``tensor_sha256`` and
    ``tensor_schema``) are ignored, so an older pointer stays a hit.
    """
    return isinstance(stored, Mapping) and all(
        stored.get(key) == value for key, value in current.items()
    )


def _pointer_matches(
    sidecar: Any, src_model_id: str, tgt_model_id: str,
    source_identity: Mapping[str, Any], target_identity: Mapping[str, Any],
) -> bool:
    return (
        isinstance(sidecar, dict)
        and sidecar.get("format_version") == _ALIGNMENT_CACHE_FORMAT_VERSION
        and sidecar.get("method") == _ALIGNMENT_METHOD
        and sidecar.get("source_model_id") == src_model_id
        and sidecar.get("target_model_id") == tgt_model_id
        and _identity_matches(sidecar.get("source_neutral_identity"), source_identity)
        and _identity_matches(sidecar.get("target_neutral_identity"), target_identity)
    )


@contextmanager
def alignment_fit_lock(src_model_id: str, tgt_model_id: str) -> Iterator[None]:
    """Single-flight a complete directional alignment fit, including loads."""
    ts_path, _ = _alignment_anchor_paths(src_model_id, tgt_model_id)
    with fit_lock(ts_path):
        yield


def save_alignment_map(
    M: AlignmentMap,
    src_model_id: str,
    tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    quality_per_layer: dict[int, float] | None = None,
    extend: bool = False,
) -> None:
    """Persist a fitted alignment map.

    Writes one bounded safetensors shard per layer, then atomically switches the
    JSON pointer.  ``extend`` carries the current pointer's other layers
    forward when it was fitted under the same identities.
    """
    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    anchor.parent.mkdir(parents=True, exist_ok=True)

    normalized = {int(idx): value for idx, value in M.items()}
    if not normalized:
        raise ValueError("cannot save an empty alignment map")
    from drowse.base.atomic import artifact_lock

    with artifact_lock(anchor):
        tensor_files: dict[str, str] = {}
        tensor_schema: dict[str, Any] = {}
        merged_quality: dict[str, float | None] = {}
        if extend and sc_path.exists():
            try:
                with open(sc_path) as handle:
                    current = json.load(handle)
                if _pointer_matches(
                    current, src_model_id, tgt_model_id,
                    source_identity, target_identity,
                ):
                    current_layers = sorted(
                        int(layer) for layer in current.get("shared_layers", [])
                    )
                    current_paths = _alignment_shard_paths(
                        anchor, current, current_layers,
                    )
                    current_schema = current.get("tensor_schema") or {}
                    # The immutable generation carries forward by name; a
                    # damaged carried layer is skipped at load and refits.
                    for layer in current_layers:
                        if current_paths[layer].exists():
                            tensor_files[str(layer)] = current_paths[layer].name
                            if isinstance(current_schema.get(str(layer)), Mapping):
                                tensor_schema[str(layer)] = dict(current_schema[str(layer)])
                    raw_quality = current.get("quality_per_layer") or {}
                    if isinstance(raw_quality, Mapping):
                        merged_quality.update({
                            str(layer): float(value)
                            for layer, value in raw_quality.items()
                            if value is not None and str(layer) in tensor_files
                        })
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                log.warning("could not extend current alignment generation: %s", exc)
                tensor_files.clear()
                tensor_schema.clear()
                merged_quality.clear()
        created: list[Path] = []
        try:
            for idx, alignment in sorted(normalized.items()):
                path = generation_path(anchor, idx)
                payload = save_safetensors({
                    "left": alignment.left.contiguous().cpu(),
                    "right": alignment.right.contiguous().cpu(),
                    "offset": alignment.offset.contiguous().cpu(),
                })
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
                tensor_schema[str(idx)] = {
                    "left": list(alignment.left.shape),
                    "right": list(alignment.right.shape),
                    "offset": list(alignment.offset.shape),
                }
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "format_version": _ALIGNMENT_CACHE_FORMAT_VERSION,
            "method": _ALIGNMENT_METHOD,
            "source_model_id": src_model_id,
            "target_model_id": tgt_model_id,
            "shared_layers": sorted(int(layer) for layer in tensor_files),
            "tensor_schema": tensor_schema,
            "source_neutral_identity": source_identity,
            "target_neutral_identity": target_identity,
            "tensor_files": tensor_files,
            "quality_per_layer": {},
        }
        if quality_per_layer:
            merged_quality.update({
                str(layer): round(float(q), 6)
                for layer, q in quality_per_layer.items()
            })
        for layer in sidecar["shared_layers"]:
            merged_quality.setdefault(str(layer), None)
        sidecar["quality_per_layer"] = merged_quality
        try:
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer is now authoritative. If directory fsync itself fails,
        # preserve the generations it names; deleting them here would turn a
        # successfully replaced pointer into an immediately broken cache.
        fsync_directory(sc_path.parent)
        _cleanup_alignment_generations(anchor, sidecar)
        fsync_directory(sc_path.parent)


def load_alignment_map(
    src_model_id: str, tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, LayerAlignment], dict[str, Any]] | None:
    """Load all or selected factor shards after the identity check.

    ``requested_layers`` materializes only the requested intersection with the
    cache's shared layers.  A layer whose shard is missing, malformed or
    non-finite is skipped (and refits), not fatal.  The returned sidecar
    remains the complete pointer.
    """

    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    from drowse.base.atomic import artifact_lock

    with artifact_lock(anchor):
        if not sc_path.exists():
            return None
        try:
            with open(sc_path) as f:
                sidecar = json.load(f)
            # Identity mismatch is the common stale-cache case.  Reject it before
            # materializing a potentially multi-GiB payload.
            if not _pointer_matches(
                sidecar, src_model_id, tgt_model_id, source_identity, target_identity,
            ):
                return None
            all_layers = sidecar.get("shared_layers")
            if (
                not isinstance(all_layers, list)
                or not all_layers
                or any(
                    isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                    for layer in all_layers
                )
            ):
                return None
            quality = sidecar.get("quality_per_layer")
            if not isinstance(quality, dict) or any(
                value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                )
                for value in quality.values()
            ):
                sidecar["quality_per_layer"] = {}
            paths = _alignment_shard_paths(anchor, sidecar, all_layers)
            selected = (
                all_layers if requested_layers is None else sorted(
                    set(all_layers) & {int(layer) for layer in requested_layers}
                )
            )
            M: dict[int, LayerAlignment] = {}
            for layer in selected:
                try:
                    tensors = load_safetensors(paths[layer].read_bytes())
                    left = tensors["left"]
                    right = tensors["right"]
                    offset = tensors["offset"]
                    if (
                        left.ndim != 2 or right.ndim != 2 or offset.ndim != 1
                        or left.shape[1] != right.shape[0]
                        or left.shape[0] != offset.shape[0]
                        or not all(
                            bool(torch.isfinite(t).all())
                            for t in (left, right, offset)
                        )
                    ):
                        continue
                    M[layer] = LayerAlignment(left, right, offset)
                except Exception:
                    # A damaged shard is one missing layer, which refits.
                    continue
            return M, sidecar
        except Exception as e:
            log.warning(
                "Corrupt alignment map cache (%s → %s): %s",
                src_model_id, tgt_model_id, e,
            )
            return None
