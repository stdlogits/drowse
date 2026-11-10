"""The ``.drowse`` manifold transport archive: codec, validation, and export.

A ``.drowse`` file is a stored ZIP whose ``pack.json`` (format v1) names one
primary manifold closure (``manifolds/<ns>/<name>/``: ``manifold.json``, the
node corpora, and every fitted tensor/sidecar pair) plus, for a templated
manifold, its template (``templates/<ns>/<name>/template.json``).  The same
wire format is read and written by the drowse.ai browser edition.

:func:`extract_drowse_archive` checks what unpacking needs (canonical
relative member paths, regular entries, the size/ratio/count bounds) and
streams the closure's entries into a caller-owned staging directory;
:func:`validate_closure` loads the staged manifold and binds its template.
Fitted tensors are validated by the tensor codec when they load
(:mod:`drowse.store.manifold_tensors`), not here.  ZIP's own per-entry CRC
covers transport corruption.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from drowse.errors import RequestError
from drowse.base.atomic import fsync_directory
from drowse.base.integrity import NAME_REGEX
from drowse.store.manifold_folder import (
    ManifoldFolder,
    _locked_manifest,
    manifold_folder_tensor_paths,
    manifold_pair_lock,
)
from drowse.store.manifold_schema import LOCAL_STATE_KEYS, ManifoldFormatError
from drowse.base.paths import manifold_dir
from drowse.store.templates import TemplateFolder, template_dir, template_lock


DROWSE_ARCHIVE_FORMAT_VERSION = 1
DROWSE_ARCHIVE_MAX_ENTRIES = 4_096
DROWSE_ARCHIVE_MAX_FILE_BYTES = 512 * 1024 * 1024
DROWSE_ARCHIVE_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO = 100
_PACK_MANIFEST_MAX_BYTES = 1024 * 1024
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

#: ``manifold.json`` keys that identify one *installed* folder generation
#: (fit transactions) or are retired (6.1's ``files`` hash map), never a
#: transported artifact.  Export drops them and install strips them from
#: whatever an archive carries.
LOCAL_ONLY_MANIFOLD_KEYS = LOCAL_STATE_KEYS


class DrowseArchiveError(RequestError):
    """A ``.drowse`` archive is malformed or unsafe."""


@dataclass(frozen=True)
class ArchiveClosure:
    """A validated manifold closure laid out under a staging directory.

    ``manifold`` / ``template`` are the closure directories; ``identity`` is
    the archive-declared ``(namespace, name)`` of the primary manifold and
    ``template_identity`` that of its template.
    """

    manifold: Path
    folder: ManifoldFolder
    identity: tuple[str, str]
    template: Path | None = None
    template_folder: TemplateFolder | None = None
    template_identity: tuple[str, str] | None = None


def _load_pack_json(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DrowseArchiveError(f"pack.json is invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise DrowseArchiveError("pack.json must be a JSON object")
    return data


def _archive_path(value: Any, *, root: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise DrowseArchiveError(f"pack.json {root!r} path must be a string")
    _validate_member_path(value)
    parts = PurePosixPath(value).parts
    if len(parts) != 3 or parts[0] != root:
        raise DrowseArchiveError(
            f"pack.json path {value!r} must be {root}/<namespace>/<name>"
        )
    namespace, name = parts[1], parts[2]
    for label, component in (("namespace", namespace), ("name", name)):
        if not NAME_REGEX.fullmatch(component):
            raise DrowseArchiveError(
                f"pack.json {root} {label} {component!r} must match "
                f"{NAME_REGEX.pattern}"
            )
    if root == "manifolds":
        _validate_manifold_identity(namespace, name)
    return value, namespace, name


def _validate_manifold_identity(namespace: str, name: str) -> None:
    from drowse.store.manifold_authoring import _validate_ns_name

    try:
        _validate_ns_name(namespace, name)
    except ManifoldFormatError as exc:
        raise DrowseArchiveError(str(exc)) from exc


def _validate_member_path(name: str) -> None:
    if not name or "\x00" in name:
        raise DrowseArchiveError("archive contains a blank or NUL-containing path")
    if "\\" in name:
        raise DrowseArchiveError(f"archive path {name!r} contains a backslash")
    if name.startswith("/") or _DRIVE_RE.match(name):
        raise DrowseArchiveError(f"archive path {name!r} is absolute")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise DrowseArchiveError(f"archive path {name!r} is not canonical")
    if PurePosixPath(name).is_absolute():
        raise DrowseArchiveError(f"archive path {name!r} is absolute")


def _validate_zip_infos(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    if not infos:
        raise DrowseArchiveError("archive is empty")
    if len(infos) > DROWSE_ARCHIVE_MAX_ENTRIES:
        raise DrowseArchiveError(
            f"archive has {len(infos)} entries; maximum is "
            f"{DROWSE_ARCHIVE_MAX_ENTRIES}"
        )
    by_name: dict[str, zipfile.ZipInfo] = {}
    casefolded: dict[str, str] = {}
    total = 0
    for info in infos:
        name = info.filename
        original = getattr(info, "orig_filename", name)
        if original != name:
            raise DrowseArchiveError(
                f"archive raw path {original!r} is not represented exactly"
            )
        _validate_member_path(original)
        _validate_member_path(name)
        if name in by_name:
            raise DrowseArchiveError(f"archive contains duplicate path {name!r}")
        folded = name.casefold()
        if folded in casefolded:
            raise DrowseArchiveError(
                f"archive paths {casefolded[folded]!r} and {name!r} collide "
                "case-insensitively"
            )
        if info.is_dir():
            raise DrowseArchiveError(f"archive entry {name!r} must be a regular file")
        if info.flag_bits & 0x1:
            raise DrowseArchiveError(f"archive entry {name!r} is encrypted")
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG):
            raise DrowseArchiveError(f"archive entry {name!r} is not a regular file")
        if info.file_size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
            raise DrowseArchiveError(
                f"archive entry {name!r} is larger than 512 MiB"
            )
        total += info.file_size
        if total > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
            raise DrowseArchiveError("archive expands beyond the 2 GiB total limit")
        if info.file_size and (
            info.compress_size == 0
            or info.file_size / info.compress_size
            > DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO
        ):
            raise DrowseArchiveError(
                f"archive entry {name!r} exceeds the 100:1 compression limit"
            )
        by_name[name] = info
        casefolded[folded] = name
    return by_name


def _validate_pack_manifest(
    data: dict[str, Any],
) -> tuple[str, tuple[str, str], str | None, tuple[str, str] | None]:
    """Read what extraction needs from ``pack.json``; ignore everything else.

    Requires ``format_version`` 1, ``kind`` ``drowse-manifold`` and the
    ``primary`` closure path; ``template`` is optional.  Producer, source and
    the per-entry ``files`` map 6.1 wrote are provenance and go unread.
    """
    version = data.get("format_version")
    if isinstance(version, bool) or version != DROWSE_ARCHIVE_FORMAT_VERSION:
        raise DrowseArchiveError(
            f"pack.json format_version must be {DROWSE_ARCHIVE_FORMAT_VERSION}"
        )
    if data.get("kind") != "drowse-manifold":
        raise DrowseArchiveError("pack.json kind must be 'drowse-manifold'")
    primary, primary_ns, primary_name = _archive_path(
        data.get("primary"), root="manifolds",
    )
    template_value = data.get("template")
    template: str | None = None
    template_identity: tuple[str, str] | None = None
    if template_value is not None:
        template, template_ns, template_name = _archive_path(
            template_value, root="templates",
        )
        template_identity = (template_ns, template_name)
    return primary, (primary_ns, primary_name), template, template_identity


def _source_provenance(source: str) -> dict[str, str | None]:
    """The ``pack.json`` ``source`` block an export writes for ``source``."""
    repository = None
    revision = None
    if source.startswith("hf://"):
        from drowse.base.hf import split_revision

        repository, revision = split_revision(source[len("hf://"):])
    return {"uri": source, "repository": repository, "revision": revision}


def _stream_entry(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, output: Path,
) -> None:
    """Copy one entry to ``output``, bounded by its declared size.

    ``zipfile`` checks the entry's CRC as the read completes.  Durability is
    the promoting caller's single pass over the finished stage
    (:func:`drowse.base.staging.promote`), not one fsync per entry.
    """
    count = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, open(output, "xb") as handle:
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > info.file_size:
                    raise DrowseArchiveError(
                        f"archive entry {info.filename!r} exceeds its declared size"
                    )
                handle.write(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DrowseArchiveError(
            f"could not read archive entry {info.filename!r} ({exc})"
        ) from exc


def extract_drowse_archive(archive_path: Path, root: Path) -> ArchiveClosure:
    """Check a ``.drowse`` and extract its closure under ``root``.

    Only the primary manifold's entries and the declared template's
    ``template.json`` are extracted; any other entry is ignored.  The
    extracted manifold directory doubles as the install stage.
    """
    source = Path(archive_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Drowse archive not found: {source}")
    if source.suffix.lower() != ".drowse":
        raise DrowseArchiveError("archive path must end in .drowse")
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise DrowseArchiveError(f"could not open {source} ({exc})") from exc
    with archive:
        infos = _validate_zip_infos(archive.infolist())
        pack_info = infos.get("pack.json")
        if pack_info is None:
            raise DrowseArchiveError("archive has no pack.json")
        if pack_info.file_size > _PACK_MANIFEST_MAX_BYTES:
            raise DrowseArchiveError("pack.json is larger than 1 MiB")
        try:
            raw_pack = archive.read(pack_info)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise DrowseArchiveError(f"could not read pack.json ({exc})") from exc
        primary, identity, template, template_identity = _validate_pack_manifest(
            _load_pack_json(raw_pack),
        )
        primary_prefix = primary + "/"
        template_entry = None if template is None else template + "/template.json"
        for name, info in infos.items():
            if name.startswith(primary_prefix) or name == template_entry:
                output = root.joinpath(*PurePosixPath(name).parts)
                _stream_entry(archive, info, output)
    manifold_path = root.joinpath(*PurePosixPath(primary).parts)
    if not (manifold_path / "manifold.json").is_file():
        raise DrowseArchiveError("archive closure has no manifold.json")
    template_path = (
        root.joinpath(*PurePosixPath(template).parts)
        if template is not None else None
    )
    folder, loaded_template = validate_closure(
        manifold_path,
        expected_name=identity[1],
        template_path=template_path,
        template_identity=template_identity,
    )
    return ArchiveClosure(
        manifold=manifold_path,
        folder=folder,
        identity=identity,
        template=template_path,
        template_folder=loaded_template,
        template_identity=template_identity,
    )


def validate_closure(
    manifold_path: Path,
    *,
    expected_name: str,
    template_path: Path | None,
    template_identity: tuple[str, str] | None,
) -> tuple[ManifoldFolder, TemplateFolder | None]:
    """Load a staged manifold (+ template) closure and bind the two.

    The manifold must load and carry ``expected_name``; a templated manifold
    needs its template closure, whose identity must match ``template_ref``.
    """
    try:
        folder = ManifoldFolder.load(manifold_path)
    except (ManifoldFormatError, OSError, ValueError) as exc:
        raise DrowseArchiveError(f"manifold closure failed validation ({exc})") from exc
    if folder.name != expected_name:
        raise DrowseArchiveError(
            f"manifold closure {expected_name!r} contains manifold identity "
            f"{folder.name!r}"
        )
    if folder.template_ref is None:
        return folder, None
    if template_path is None or template_identity is None:
        raise DrowseArchiveError("templated manifold is missing its template closure")
    expected_ref = "/".join(template_identity)
    if folder.template_ref != expected_ref:
        raise DrowseArchiveError(
            f"manifold template_ref {folder.template_ref!r} does not match "
            f"closure {expected_ref!r}"
        )
    try:
        loaded_template = TemplateFolder.load(template_path)
    except (OSError, ValueError) as exc:
        raise DrowseArchiveError(f"closure template failed validation ({exc})") from exc
    if loaded_template.name != template_identity[1]:
        raise DrowseArchiveError(
            f"template closure {expected_ref!r} contains identity "
            f"{loaded_template.name!r}"
        )
    return folder, loaded_template


def _fitted_tensor_paths(folder: ManifoldFolder) -> list[Path]:
    """The folder's published fits with a current, readable sidecar."""
    tensors = [folder.tensor_path(stem) for stem in folder.tensor_models()]
    if folder.fit_mode == "baked" and not tensors:
        raise DrowseArchiveError("baked manifold closure has no fitted tensor")
    return sorted(tensors)


def _payload_paths(folder: ManifoldFolder) -> list[Path]:
    """``manifold.json``, the node corpora, and every published fitted pair."""
    paths = [folder.folder / "manifold.json"]
    if folder.fit_mode != "baked":
        paths.extend(folder.node_path(i) for i in range(len(folder.node_labels)))
    for tensor in _fitted_tensor_paths(folder):
        paths.extend((tensor, tensor.with_suffix(".json")))
    return paths


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_zip_path(archive: zipfile.ZipFile, archive_path: str, path: Path) -> None:
    with archive.open(_zip_info(archive_path), "w") as target, open(path, "rb") as source:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _template_ref_parts(ref: str) -> tuple[str, str]:
    parts = ref.split("/")
    if len(parts) != 2 or any(not NAME_REGEX.fullmatch(part) for part in parts):
        raise DrowseArchiveError(
            "Drowse archive requires a namespace-qualified template_ref"
        )
    return parts[0], parts[1]


def _resolve_transport_template(folder: ManifoldFolder) -> TemplateFolder | None:
    from drowse.store.templates import resolve_template

    if folder.template_ref is None:
        return None
    _template_ref_parts(folder.template_ref)
    loaded = resolve_template(folder.template_ref)
    if loaded.path is None:
        raise DrowseArchiveError("referenced template has no on-disk path")
    return loaded


@contextmanager
def _source_closure_locks(folder_path: Path) -> Iterator[None]:
    """Freeze one installed manifold (+ its template) for a consistent read.

    Global order: manifest lock, then the template lock, then every fitted
    pair lock.  The template is computable from the locked manifest because
    transportable ``template_ref`` values are namespace-qualified.
    """
    with ExitStack() as stack:
        stack.enter_context(_locked_manifest(folder_path))
        ref = ManifoldFolder.load(folder_path).template_ref
        if ref is not None:
            stack.enter_context(template_lock(template_dir(*_template_ref_parts(ref))))
        for tensor in manifold_folder_tensor_paths(folder_path):
            stack.enter_context(manifold_pair_lock(tensor))
        yield


def _transport_manifest_bytes(folder: ManifoldFolder) -> bytes:
    """``manifold.json`` as transported, local-only keys dropped."""
    payload = json.loads((folder.folder / "manifold.json").read_bytes())
    for key in LOCAL_ONLY_MANIFOLD_KEYS:
        payload.pop(key, None)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def export_drowse_archive(
    namespace: str,
    name: str,
    *,
    output: str | Path | None = None,
    include_tensor: Callable[[str], bool] | None = None,
) -> Path:
    """Export one installed manifold and its template.

    ``output`` is a ``.drowse`` path or a directory (default: the working
    directory), receiving ``<name>.drowse``.  ``include_tensor`` narrows the
    exported fitted pairs by tensor filename (``drowse pack push``'s model /
    variant filter).
    """
    from drowse import __version__

    _validate_manifold_identity(namespace, name)
    folder_path = manifold_dir(namespace, name)
    if not (folder_path / "manifold.json").is_file():
        raise FileNotFoundError(f"manifold {namespace}/{name} not found at {folder_path}")
    destination = (
        Path(output).expanduser()
        if output is not None else Path.cwd() / f"{name}.drowse"
    )
    if destination.exists() and destination.is_dir():
        destination = destination / f"{name}.drowse"
    if destination.suffix.lower() != ".drowse":
        raise DrowseArchiveError("export path must end in .drowse")
    if destination.resolve(strict=False).is_relative_to(
        folder_path.resolve(strict=False)
    ):
        raise DrowseArchiveError(
            "Drowse archive export path must be outside its manifold closure"
        )

    with _source_closure_locks(folder_path):
        folder = ManifoldFolder.load(folder_path)
        if folder.name != name:
            raise DrowseArchiveError(
                f"manifold path {namespace}/{name} contains identity "
                f"{folder.name!r}"
            )
        template_folder = _resolve_transport_template(folder)
        template_path = (
            template_folder.path if template_folder is not None else None
        )
        if template_path is not None and destination.resolve(
            strict=False,
        ).is_relative_to(template_path.resolve(strict=False)):
            raise DrowseArchiveError(
                "Drowse archive export path must be outside its template closure"
            )
        manifest_path = folder.folder / "manifold.json"

        def kept(path: Path) -> bool:
            if include_tensor is None or path.parent != folder.folder:
                return True
            return path == manifest_path or include_tensor(
                path.with_suffix(".safetensors").name,
            )

        primary = f"manifolds/{namespace}/{name}"
        payloads: list[tuple[str, Path | bytes]] = [
            (
                f"{primary}/{path.relative_to(folder.folder).as_posix()}",
                _transport_manifest_bytes(folder) if path == manifest_path else path,
            )
            for path in _payload_paths(folder) if kept(path)
        ]
        template_key = None
        if template_folder is not None and template_path is not None:
            template_key = f"templates/{template_path.parent.name}/{template_folder.name}"
            payloads.append((f"{template_key}/template.json", template_path / "template.json"))
        _write_archive(
            destination,
            payloads,
            primary=primary,
            template=template_key,
            source=_source_provenance(folder.source or "local"),
            producer_version=__version__,
        )
    return destination


def _write_archive(
    destination: Path,
    payloads: list[tuple[str, Path | bytes]],
    *,
    primary: str,
    template: str | None,
    source: dict[str, str | None],
    producer_version: str,
) -> None:
    if len(payloads) + 1 > DROWSE_ARCHIVE_MAX_ENTRIES:
        raise DrowseArchiveError(
            f"Drowse archive would exceed {DROWSE_ARCHIVE_MAX_ENTRIES} entries"
        )
    payload_bytes = 0
    for archive_path, item in payloads:
        _validate_member_path(archive_path)
        size = len(item) if isinstance(item, bytes) else item.stat().st_size
        if size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
            raise DrowseArchiveError(
                f"archive payload {archive_path!r} is larger than 512 MiB"
            )
        payload_bytes += size
    manifest = {
        "format_version": DROWSE_ARCHIVE_FORMAT_VERSION,
        "kind": "drowse-manifold",
        "producer": {"name": "drowse", "version": producer_version},
        "primary": primary,
        "template": template,
        "source": source,
    }
    raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if len(raw_manifest) > _PACK_MANIFEST_MAX_BYTES:
        raise DrowseArchiveError("pack.json would be larger than 1 MiB")
    if payload_bytes + len(raw_manifest) > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
        raise DrowseArchiveError("Drowse archive would expand beyond 2 GiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info("pack.json"), raw_manifest)
            for archive_path, item in sorted(payloads, key=lambda pair: pair[0]):
                if isinstance(item, bytes):
                    archive.writestr(_zip_info(archive_path), item)
                else:
                    _write_zip_path(archive, archive_path, item)
        with open(temp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)
