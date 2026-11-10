"""The neuron-lens catalog: labels and firing tables for a model's units.

`stdlogits/neuron-lens <https://huggingface.co/datasets/stdlogits/neuron-lens>`_
catalogues every neuron pole and head of Qwen3-8B and Gemma 4 12B (base):
weight reads, activation statistics, legibility tests and labels written by
Claude Haiku 5.5.  ``unit fetch`` binds it for one of those models at a
pinned commit, copying only what a unit view shows:

- ``neurons/labels/L<LL>.parquet`` and ``heads/labels.parquet`` whole (the
  labels with their confidences and notes);
- ``neurons/acts`` and ``heads/acts`` projected to their scalar columns (the
  firing rate, ``act_top``, quantiles; never the top contexts);
- ``neurons/legible`` and ``heads/legible`` projected to the legibility
  tests' scalars (which reads hold at the unit's own top firings);
- ``token_counts.pt``'s total counts (every vocabulary id's occurrences over
  the activation corpora, 88M tokens on Qwen3-8B), stored as
  ``token_counts.safetensors``: the S read's token mask keeps the ids seen
  :data:`~drowse.lens.footprint.MIN_TOKEN_COUNT` or more times, as
  neuron-lens's ``kept`` does.

The catalog's ``act_top`` per pole (``out_top`` per head) is also the units'
reading scale (:meth:`UnitCatalog.reading_scales`).

The bulk (``reads``, ``drivers``, media, the causal sample) is never
downloaded: drowse computes its own reads.  Projection reads only the wanted
column chunks over HTTP ranges.  The binding lives under
``$DROWSE_HOME/models/<id>/units/catalog/`` with ``catalog.json``
(:data:`CATALOG_FORMAT`) recording the repository, the commit, and per file
its upstream path, the columns kept and the row count.
A binding of another format, repository or commit is not read (``fetch``
again); there is no migration.

Labels are data, shown with :data:`LABEL_CREDIT` and the unit's page on
neuron-lens.com (:func:`unit_url`).  Reading and fetching need ``pyarrow``
(the ``drowse.ai[hf]`` extra).
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.paths import model_dir
from drowse.base.staging import promote, recover_backup, staging_area
from drowse.errors import DrowseError
from drowse.naming import HeadRef, NeuronRef, UnitRef

if TYPE_CHECKING:
    import torch

    from drowse.lens.footprint import TokenCounts
    from drowse.units.stats import UnitScales

#: The catalog repository and the commit every binding pins.
CATALOG_REPO = "stdlogits/neuron-lens"
CATALOG_REVISION = "b5bf11a2b9e35c9100a0fa0995ff7a4e57f5b777"
#: ``catalog.json``'s format; another is refused (fetch again).  2: the
#: binding holds the token counts.
CATALOG_FORMAT = 2
#: The catalog's token counts, upstream (a torch pickle of plain tensors,
#: loaded ``weights_only``) and in the binding.
TOKEN_COUNTS_UPSTREAM = "token_counts.pt"
TOKEN_COUNTS_FILE = "token_counts.safetensors"
#: The explorer every unit links to.
SITE = "https://neuron-lens.com"
LABEL_CREDIT = "Labels by Claude Haiku 5.5"


@dataclass(frozen=True)
class CatalogModel:
    """One catalogued model: its Hugging Face id and the catalog's slug (the
    dataset's top-level folder and neuron-lens.com's path)."""

    model_id: str
    slug: str


#: The catalogued models, by lower-cased model id.  Gemma 4 12B's catalog is
#: of the base model; other checkpoints (an instruction-tuned one) have their
#: own neurons and get no catalog.
CATALOG_MODELS: Mapping[str, CatalogModel] = {
    "qwen/qwen3-8b": CatalogModel("Qwen/Qwen3-8B", "qwen3-8b"),
    "google/gemma-4-12b": CatalogModel("google/gemma-4-12B", "gemma-4-12b"),
}

#: Columns kept per table (``None``: every column).  Keys first.
NEURON_KEYS = ("layer", "neuron", "pole")
HEAD_KEYS = ("layer", "head", "kv_head")
NEURON_COLUMNS: Mapping[str, tuple[str, ...] | None] = {
    "labels": None,
    "acts": NEURON_KEYS + (
        "act_top", "act_max", "pos_share", "rate", "rate_dclm", "rate_zh",
        "rate_code", "rate_chat", "q50", "q90", "q99", "q999", "mean", "std",
    ),
    "legible": NEURON_KEYS + (
        "silent", "dark", "write_R_q", "write_S_q", "read_R_q", "read_S_q",
    ),
}
HEAD_COLUMNS: Mapping[str, tuple[str, ...] | None] = {
    "labels": None,
    "acts": HEAD_KEYS + (
        "out_top", "out_max", "rate", "rate_dclm", "rate_zh", "rate_code",
        "rate_chat", "q50", "q90", "q99", "q999", "mean", "std", "sink",
        "prev", "self", "induction", "duplicate",
    ),
    "legible": HEAD_KEYS + (
        "ov_R_q", "ov_copy_q", "ov_S_q", "qk_S_q", "pattern", "dark",
    ),
}
_LABEL_FIELDS = ("fires", "fires_conf", "attends", "attends_conf", "writes", "writes_conf", "note")


class UnitCatalogError(RuntimeError, DrowseError):
    """The catalog could not be fetched or read (500 unless narrowed)."""


class NoUnitCatalogError(LookupError, DrowseError):
    """The model has no neuron-lens catalog, or it is not fetched (404)."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self))


class UnitCatalogDependencyError(ImportError, DrowseError):
    """``pyarrow`` is missing (503: the ``drowse.ai[hf]`` extra)."""

    def user_message(self) -> tuple[int, str]:
        return (503, str(self))


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise UnitCatalogDependencyError(
            "the unit catalog needs pyarrow: pip install 'drowse.ai[hf]'"
        ) from exc
    return pa, pq


def catalog_model(model_id: str) -> CatalogModel | None:
    """The catalogued model ``model_id`` names (case-insensitively), or ``None``."""
    return CATALOG_MODELS.get(str(model_id).strip().lower())


def require_catalog_model(model_id: str) -> CatalogModel:
    entry = catalog_model(model_id)
    if entry is None:
        names = ", ".join(model.model_id for model in CATALOG_MODELS.values())
        raise NoUnitCatalogError(
            f"no neuron-lens catalog for {model_id} (catalogued: {names})"
        )
    return entry


def catalog_dir(model_id: str) -> Path:
    """``$DROWSE_HOME/models/<safe id>/units/catalog``."""
    return model_dir(model_id) / "units" / "catalog"


def unit_url(slug: str, ref: UnitRef) -> str:
    """The unit's neuron-lens.com page: ``/<slug>/L<L>/n<n>`` or ``/h<h>``."""
    tail = f"n{ref.index}" if isinstance(ref, NeuronRef) else f"h{ref.index}"
    return f"{SITE}/{slug}/L{int(ref.layer)}/{tail}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _remote_files(slug: str, revision: str) -> list[str]:
    """The catalog file paths a binding copies (one listing request)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(CATALOG_REPO, revision=revision)
    out: list[str] = []
    for sibling in info.siblings or ():
        path = sibling.rfilename
        parts = path.split("/")
        if parts[0] != slug:
            continue
        counts = len(parts) == 2 and parts[1] == TOKEN_COUNTS_UPSTREAM
        if not counts and not path.endswith(".parquet"):
            continue
        wanted = counts or (
            len(parts) == 4 and parts[1] == "neurons" and parts[2] in NEURON_COLUMNS
        ) or (
            len(parts) == 3 and parts[1] == "heads" and parts[2][: -len(".parquet")] in HEAD_COLUMNS
        )
        if wanted:
            out.append(path)
    return sorted(out)


def _read_remote_table(path: str, revision: str, columns: Sequence[str] | None) -> Any:
    """``path`` of the catalog at ``revision`` as a pyarrow table, reading
    only ``columns`` (range requests over the Hub filesystem)."""
    _pa, pq = _arrow()
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return pq.read_table(
        f"datasets/{CATALOG_REPO}@{revision}/{path}",
        columns=None if columns is None else list(columns),
        filesystem=fs,
    )


def _columns_for(path: str) -> tuple[str, ...] | None:
    parts = path.split("/")
    if parts[1] == "neurons":
        return NEURON_COLUMNS[parts[2]]
    return HEAD_COLUMNS[parts[2][: -len(".parquet")]]


def _download_remote(path: str, revision: str) -> Path:
    """``path`` of the catalog at ``revision``, downloaded whole (the Hub
    cache)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        CATALOG_REPO, path, repo_type="dataset", revision=revision,
    ))


def _write_token_counts(source: Path, destination: Path) -> int:
    """Copy ``token_counts.pt``'s total counts (``[vocabulary]`` int64) into
    ``destination`` as safetensors; returns the vocabulary size.  The pickle
    is loaded ``weights_only`` (plain tensors and strings only)."""
    import torch
    from safetensors.torch import save_file

    record = torch.load(source, map_location="cpu", weights_only=True)
    counts = record.get("counts") if isinstance(record, Mapping) else None
    if not isinstance(counts, torch.Tensor) or counts.ndim != 1:
        raise UnitCatalogError(f"{source.name} holds no [vocabulary] counts tensor")
    save_file({"counts": counts.to(torch.int64).contiguous()}, str(destination))
    return int(counts.numel())


def fetch_catalog(
    model_id: str,
    *,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Bind the neuron-lens catalog for ``model_id`` at :data:`CATALOG_REVISION`.

    Raises :class:`NoUnitCatalogError` (404) for an uncatalogued model.  An
    existing binding at this commit and format is kept unless ``force``.
    Returns the binding record (``catalog.json``).  Network and disk only:
    no model work, no lock beyond the binding's own artifact lock.
    """
    entry = require_catalog_model(model_id)
    _pa, pq = _arrow()
    report = on_progress or (lambda _message: None)
    target = catalog_dir(model_id)
    with artifact_lock(target):
        recover_backup(target)
        existing = _read_binding(target)
        if existing is not None and not force and _current(existing, entry):
            report(f"already bound: {CATALOG_REPO}@{CATALOG_REVISION[:12]} ({entry.slug})")
            return existing
        report(f"listing {CATALOG_REPO}@{CATALOG_REVISION[:12]} ({entry.slug})…")
        remote = _remote_files(entry.slug, CATALOG_REVISION)
        if not any(path.endswith("labels.parquet") for path in remote):
            raise UnitCatalogError(
                f"{CATALOG_REPO}@{CATALOG_REVISION[:12]} holds no {entry.slug} labels"
            )
        files: dict[str, dict[str, Any]] = {}
        with staging_area() as stage:
            root = stage / "catalog"
            if not any(path.endswith("/" + TOKEN_COUNTS_UPSTREAM) for path in remote):
                raise UnitCatalogError(
                    f"{CATALOG_REPO}@{CATALOG_REVISION[:12]} holds no {entry.slug} "
                    f"{TOKEN_COUNTS_UPSTREAM}"
                )
            for done, path in enumerate(remote, start=1):
                if path.endswith("/" + TOKEN_COUNTS_UPSTREAM):
                    root.mkdir(parents=True, exist_ok=True)
                    destination = root / TOKEN_COUNTS_FILE
                    rows = _write_token_counts(
                        _download_remote(path, CATALOG_REVISION), destination,
                    )
                    files[TOKEN_COUNTS_FILE] = {
                        "upstream": path,
                        "columns": ["counts"],
                        "rows": rows,
                    }
                    report(f"{done}/{len(remote)} {path} ({rows} token ids)")
                    continue
                local = path.split("/", 1)[1]
                columns = _columns_for(path)
                table = _read_remote_table(path, CATALOG_REVISION, columns)
                destination = root / local
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, str(destination), compression="zstd")
                files[local] = {
                    "upstream": path,
                    "columns": None if columns is None else list(columns),
                    "rows": int(table.num_rows),
                }
                report(f"{done}/{len(remote)} {path} ({table.num_rows} rows)")
            record = {
                "format": CATALOG_FORMAT,
                "model_id": entry.model_id,
                "slug": entry.slug,
                "repo": CATALOG_REPO,
                "revision": CATALOG_REVISION,
                "credit": LABEL_CREDIT,
                "site": f"{SITE}/{entry.slug}",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": files,
            }
            write_json_atomic(root / "catalog.json", record)
            promote(root, target)
        report(f"bound {len(files)} tables from {CATALOG_REPO}@{CATALOG_REVISION[:12]}")
        return record


def _read_binding(target: Path) -> dict[str, Any] | None:
    path = target / "catalog.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _current(record: Mapping[str, Any], entry: CatalogModel) -> bool:
    return (
        record.get("format") == CATALOG_FORMAT
        and record.get("repo") == CATALOG_REPO
        and record.get("revision") == CATALOG_REVISION
        and record.get("slug") == entry.slug
    )


def remove_catalog(model_id: str) -> bool:
    """Drop the model's binding; whether there was one."""
    import shutil

    target = catalog_dir(model_id)
    with artifact_lock(target):
        recover_backup(target)
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True


def catalog_status(model_id: str) -> dict[str, Any]:
    """What a unit view or ``unit ls`` says about the binding.

    ``available``: the model is catalogued; ``bound``: a current binding is
    on disk; ``reason``: why labels are absent otherwise.  Disk only.
    """
    entry = catalog_model(model_id)
    if entry is None:
        names = ", ".join(model.model_id for model in CATALOG_MODELS.values())
        return {
            "model_id": model_id, "available": False, "bound": False, "slug": None,
            "repo": CATALOG_REPO, "revision": CATALOG_REVISION, "credit": None,
            "reason": f"no neuron-lens catalog for this model (catalogued: {names})",
            "binding": None,
        }
    record = _read_binding(catalog_dir(model_id))
    bound = record is not None and _current(record, entry)
    reason = None
    if record is None:
        reason = f"the catalog is not fetched (`drowse unit fetch {entry.model_id}`)"
    elif not bound:
        reason = (
            f"the bound catalog is not {CATALOG_REPO}@{CATALOG_REVISION[:12]} "
            f"format {CATALOG_FORMAT} (`drowse unit fetch {entry.model_id} -f`)"
        )
    return {
        "model_id": model_id, "available": True, "bound": bound, "slug": entry.slug,
        "repo": CATALOG_REPO, "revision": CATALOG_REVISION,
        "credit": LABEL_CREDIT, "reason": reason,
        "binding": None if record is None else {
            "revision": record.get("revision"),
            "fetched_at": record.get("fetched_at"),
            "files": len(record.get("files") or {}),
        },
    }


def list_catalogs() -> list[dict[str, Any]]:
    """Every catalogued model's status."""
    return [catalog_status(model.model_id) for model in CATALOG_MODELS.values()]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if value == value else None
    return value


class UnitCatalog:
    """A bound catalog on disk, read a layer at a time (the latest few kept)."""

    def __init__(self, root: Path, record: Mapping[str, Any], *, resident: int = 12) -> None:
        self.root = root
        self.record = dict(record)
        self.slug = str(record["slug"])
        self._resident = int(resident)
        self._tables: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def open(cls, model_id: str) -> "UnitCatalog | None":
        """The current binding for ``model_id``, or ``None``."""
        entry = catalog_model(model_id)
        if entry is None:
            return None
        target = catalog_dir(model_id)
        record = _read_binding(target)
        if record is None or not _current(record, entry):
            return None
        _arrow()
        return cls(target, record)

    def _table(self, local: str) -> Any | None:
        with self._lock:
            hit = self._tables.get(local)
        if hit is not None:
            return hit
        path = self.root / local
        if not path.is_file():
            return None
        _pa, pq = _arrow()
        table = pq.read_table(str(path))
        with self._lock:
            if len(self._tables) >= self._resident:
                self._tables.pop(next(iter(self._tables)))
            self._tables[local] = table
        return table

    def _neuron_row(self, table: str, layer: int, n: int, pole: int) -> dict[str, Any] | None:
        data = self._table(f"neurons/{table}/L{int(layer):02d}.parquet")
        if data is None:
            return None
        f = int(data.num_rows) // 2
        if not 0 <= int(n) < f:
            return None
        row = int(n) if pole > 0 else f + int(n)
        return {k: _clean(v) for k, v in data.slice(row, 1).to_pylist()[0].items()}

    def _head_row(self, table: str, layer: int, h: int) -> dict[str, Any] | None:
        data = self._table(f"heads/{table}.parquet")
        if data is None:
            return None
        layers = data.column("layer")
        n_layers = int(max(layers.to_pylist())) + 1 if data.num_rows else 0
        if n_layers == 0:
            return None
        heads = int(data.num_rows) // n_layers
        if not (0 <= int(layer) < n_layers and 0 <= int(h) < heads):
            return None
        row = int(layer) * heads + int(h)
        return {k: _clean(v) for k, v in data.slice(row, 1).to_pylist()[0].items()}

    def url(self, ref: UnitRef) -> str:
        return unit_url(self.slug, ref)

    def neuron_label(self, layer: int, n: int, pole: int) -> dict[str, Any] | None:
        """A pole's label (fires, writes, their confidences, the note), its
        legibility tests, the credit and the unit's page."""
        row = self._neuron_row("labels", layer, n, pole)
        if row is None:
            return None
        legible = self._neuron_row("legible", layer, n, pole)
        return {
            **{key: row[key] for key in _LABEL_FIELDS if key in row},
            "legible": None if legible is None else {
                key: value for key, value in legible.items() if key not in NEURON_KEYS
            },
            "credit": LABEL_CREDIT,
            "url": self.url(NeuronRef(layer=int(layer), index=int(n))),
        }

    def neuron_acts(self, layer: int, n: int, pole: int) -> dict[str, Any] | None:
        row = self._neuron_row("acts", layer, n, pole)
        if row is None:
            return None
        return {key: value for key, value in row.items() if key not in NEURON_KEYS}

    def head_label(self, layer: int, h: int) -> dict[str, Any] | None:
        row = self._head_row("labels", layer, h)
        if row is None:
            return None
        legible = self._head_row("legible", layer, h)
        return {
            **{key: row[key] for key in _LABEL_FIELDS if key in row},
            "legible": None if legible is None else {
                key: value for key, value in legible.items() if key not in HEAD_KEYS
            },
            "credit": LABEL_CREDIT,
            "url": self.url(HeadRef(layer=int(layer), index=int(h))),
        }

    def head_acts(self, layer: int, h: int) -> dict[str, Any] | None:
        row = self._head_row("acts", layer, h)
        if row is None:
            return None
        return {key: value for key, value in row.items() if key not in HEAD_KEYS}

    @property
    def identity(self) -> str:
        """``stdlogits/neuron-lens@<commit[:12]>``: what a reading scale or a
        token mask taken from this binding names."""
        return f"{self.record.get('repo', CATALOG_REPO)}@{str(self.record.get('revision', ''))[:12]}"

    @functools.cached_property
    def reading_scales(self) -> "UnitScales":
        """The ``"catalog"`` reading scales: every pole's ``act_top`` and every
        head's ``out_top`` (:func:`~drowse.units.stats.reading_scales`).  Disk
        only; read once per opened binding.  A table missing or of the wrong
        shape is a :class:`UnitCatalogError`."""
        import torch

        from drowse.units.stats import reading_scales

        _pa, pq = _arrow()
        heads_path = self.root / "heads" / "acts.parquet"
        if not heads_path.is_file():
            raise UnitCatalogError(f"the bound catalog has no {heads_path.name}")
        heads = pq.read_table(str(heads_path), columns=["layer", "head", "out_top"]).to_pydict()
        n_layers = int(max(heads["layer"])) + 1 if heads["layer"] else 0
        if n_layers == 0 or len(heads["layer"]) % n_layers:
            raise UnitCatalogError("the bound catalog's head table is not one row per head")
        per_layer = len(heads["layer"]) // n_layers
        head_top = torch.tensor(heads["out_top"], dtype=torch.float32).reshape(n_layers, per_layer)
        neuron_top: dict[int, torch.Tensor] = {}
        for layer in range(n_layers):
            path = self.root / "neurons" / "acts" / f"L{layer:02d}.parquet"
            if not path.is_file():
                raise UnitCatalogError(f"the bound catalog has no neurons/acts/{path.name}")
            table = pq.read_table(str(path), columns=["pole", "act_top"]).to_pydict()
            rows = len(table["act_top"])
            f = rows // 2
            if rows % 2 or (f and (table["pole"][0] <= 0 or table["pole"][f] >= 0)):
                raise UnitCatalogError(
                    f"the bound catalog's neurons/acts/{path.name} is not + poles then − poles"
                )
            values = torch.tensor(
                [float("nan") if v is None else float(v) for v in table["act_top"]],
                dtype=torch.float32,
            )
            neuron_top[layer] = values.nan_to_num(0.0).clamp_min(0.0).reshape(2, f)
        return reading_scales(
            neuron_top,
            {layer: head_top[layer].nan_to_num(0.0).clamp_min(0.0) for layer in range(n_layers)},
            source="catalog",
            detail=f"neuron-lens catalog {self.identity}",
        )

    @functools.cached_property
    def token_counts(self) -> "TokenCounts | None":
        """The catalog's token counts (``[vocabulary]`` occurrences over its
        activation corpora), or ``None`` when the binding holds none."""
        from safetensors.torch import load_file

        from drowse.lens.footprint import TokenCounts

        path = self.root / TOKEN_COUNTS_FILE
        if not path.is_file():
            return None
        counts = load_file(str(path))["counts"]
        return TokenCounts(
            counts=counts, source="catalog", detail=f"neuron-lens catalog {self.identity}",
        )


#: ``catalog.json``'s on-disk identity (inode, mtime, size), or ``None``
#: when there is no binding: what :class:`CatalogBinding` compares before
#: trusting what it opened.
_Identity = tuple[int, int, int] | None


def _binding_identity(model_id: str) -> _Identity:
    try:
        stat = (catalog_dir(model_id) / "catalog.json").stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


class CatalogBinding:
    """The session's view of its model's catalog, opened lazily.

    Another process may fetch, refetch or remove the binding (``drowse unit
    fetch`` beside a running ``drowse serve``), so every :meth:`get` compares
    ``catalog.json``'s on-disk identity (one ``stat``) with the one it opened
    and reopens on a change, a cached "no catalog" included.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._catalog: UnitCatalog | None = None
        self._opened = False
        self._identity: _Identity = None
        self._lock = threading.Lock()

    def get(self) -> UnitCatalog | None:
        identity = _binding_identity(self.model_id)
        with self._lock:
            if not self._opened or identity != self._identity:
                try:
                    self._catalog = UnitCatalog.open(self.model_id)
                except UnitCatalogDependencyError:
                    self._catalog = None
                self._opened = True
                self._identity = identity
            return self._catalog

    def status(self) -> dict[str, Any]:
        status = catalog_status(self.model_id)
        if status["bound"]:
            try:
                _arrow()
            except UnitCatalogDependencyError as exc:
                status = {**status, "bound": False, "reason": str(exc)}
        return status

    def view(self) -> tuple[UnitCatalog | None, dict[str, Any]]:
        """``(catalog, status)`` that agree: ``status["bound"]`` is whether
        ``catalog`` opened, and an unbound status always says why."""
        catalog = self.get()
        status = self.status()
        if catalog is not None:
            return catalog, {**status, "bound": True, "slug": catalog.slug, "reason": None}
        reason = status.get("reason") if not status["bound"] else None
        return None, {
            **status, "bound": False,
            "reason": reason or "the catalog binding changed while it was read; read again",
        }

    def reset(self) -> None:
        with self._lock:
            self._catalog = None
            self._opened = False
            self._identity = None

    def fetch(self, *, force: bool = False, on_progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        try:
            return fetch_catalog(self.model_id, force=force, on_progress=on_progress)
        finally:
            self.reset()

    def remove(self) -> bool:
        try:
            return remove_catalog(self.model_id)
        finally:
            self.reset()


__all__ = [
    "CATALOG_FORMAT",
    "CATALOG_MODELS",
    "CATALOG_REPO",
    "CATALOG_REVISION",
    "CatalogBinding",
    "CatalogModel",
    "HEAD_COLUMNS",
    "LABEL_CREDIT",
    "NEURON_COLUMNS",
    "NoUnitCatalogError",
    "SITE",
    "UnitCatalog",
    "UnitCatalogDependencyError",
    "UnitCatalogError",
    "catalog_dir",
    "catalog_model",
    "catalog_status",
    "fetch_catalog",
    "list_catalogs",
    "remove_catalog",
    "require_catalog_model",
    "unit_url",
]
