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
  tests' scalars (which reads hold at the unit's own top firings).

The bulk (``reads``, ``drivers``, media, the causal sample) is never
downloaded: drowse computes its own reads.  Projection reads only the wanted
column chunks over HTTP ranges.  The binding lives under
``$DROWSE_HOME/models/<id>/units/catalog/`` with ``catalog.json``
(:data:`CATALOG_FORMAT`) recording the repository, the commit, and per file
its upstream LFS ``sha256``, the columns kept and the local file's ``sha256``.
A binding of another format, repository or commit is not read (``fetch``
again); there is no migration.

Labels are data, shown with :data:`LABEL_CREDIT` and the unit's page on
neuron-lens.com (:func:`unit_url`).  Reading and fetching need ``pyarrow``
(the ``drowse.ai[hf]`` extra).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.integrity import hash_file
from drowse.base.paths import model_dir
from drowse.base.staging import promote, recover_backup, staging_area
from drowse.errors import DrowseError
from drowse.naming import HeadRef, NeuronRef, UnitRef

#: The catalog repository and the commit every binding pins.
CATALOG_REPO = "stdlogits/neuron-lens"
CATALOG_REVISION = "b5bf11a2b9e35c9100a0fa0995ff7a4e57f5b777"
#: ``catalog.json``'s format; another is refused (fetch again).
CATALOG_FORMAT = 1
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

@dataclass(frozen=True)
class RemoteFile:
    """One catalog file at the pinned commit: its path and LFS digest."""

    path: str
    sha256: str | None
    size: int | None


def _remote_files(slug: str, revision: str) -> list[RemoteFile]:
    """The catalog files a binding copies, with their upstream digests (one
    metadata request)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(CATALOG_REPO, revision=revision, files_metadata=True)
    out: list[RemoteFile] = []
    for sibling in info.siblings or ():
        path = sibling.rfilename
        parts = path.split("/")
        if parts[0] != slug or not path.endswith(".parquet"):
            continue
        wanted = (
            len(parts) == 4 and parts[1] == "neurons" and parts[2] in NEURON_COLUMNS
        ) or (
            len(parts) == 3 and parts[1] == "heads" and parts[2][: -len(".parquet")] in HEAD_COLUMNS
        )
        if wanted:
            lfs = getattr(sibling, "lfs", None)
            out.append(RemoteFile(
                path,
                None if lfs is None else getattr(lfs, "sha256", None),
                getattr(sibling, "size", None),
            ))
    return sorted(out, key=lambda item: item.path)


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
        if not any(item.path.endswith("labels.parquet") for item in remote):
            raise UnitCatalogError(
                f"{CATALOG_REPO}@{CATALOG_REVISION[:12]} holds no {entry.slug} labels"
            )
        files: dict[str, dict[str, Any]] = {}
        with staging_area() as stage:
            root = stage / "catalog"
            for done, item in enumerate(remote, start=1):
                local = item.path.split("/", 1)[1]
                columns = _columns_for(item.path)
                table = _read_remote_table(item.path, CATALOG_REVISION, columns)
                destination = root / local
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, str(destination), compression="zstd")
                files[local] = {
                    "upstream": item.path,
                    "upstream_sha256": item.sha256,
                    "columns": None if columns is None else list(columns),
                    "rows": int(table.num_rows),
                    "sha256": hash_file(destination),
                }
                report(f"{done}/{len(remote)} {item.path} ({table.num_rows} rows)")
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
    """What a unit view or ``unit catalog`` says about the binding.

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


class CatalogBinding:
    """The session's view of its model's catalog: opened lazily, reopened
    after a fetch or removal."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._catalog: UnitCatalog | None = None
        self._opened = False
        self._lock = threading.Lock()

    def get(self) -> UnitCatalog | None:
        with self._lock:
            if not self._opened:
                try:
                    self._catalog = UnitCatalog.open(self.model_id)
                except UnitCatalogDependencyError:
                    self._catalog = None
                self._opened = True
            return self._catalog

    def status(self) -> dict[str, Any]:
        status = catalog_status(self.model_id)
        if status["bound"]:
            try:
                _arrow()
            except UnitCatalogDependencyError as exc:
                status = {**status, "bound": False, "reason": str(exc)}
        return status

    def reset(self) -> None:
        with self._lock:
            self._catalog = None
            self._opened = False

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
    "RemoteFile",
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
