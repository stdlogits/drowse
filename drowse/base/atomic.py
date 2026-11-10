"""Atomic file writes for ``~/.drowse/`` state.

A plain ``open(p, "w") + json.dump`` corrupts the file on SIGKILL or ENOSPC —
the partial bytes remain at ``p`` with no signal to the loader. ``write_*``
helpers here use private, unique staging files in the same directory,
``flush()`` + ``fsync()``, then ``os.replace()`` the tempfile into place.
Same-dir staging is required: ``os.replace()`` is only atomic on the same
filesystem.

A crash between the write and the replace can leave a staging file orphaned;
the next loader sees the prior good ``<path>`` (or no file at all on a
first-time write). The orphan is harmless: no reader looks at staging names.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from contextlib import suppress
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary


_ARTIFACT_LOCKS_GUARD = threading.Lock()
_ARTIFACT_LOCKS: WeakValueDictionary[Path, threading.RLock] = WeakValueDictionary()
_ARTIFACT_LOCK_STATE = threading.local()


def _lock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def artifact_lock(path: Path):
    """Cross-process, thread-reentrant lock for one logical artifact pair."""
    path = Path(path).expanduser().resolve(strict=False)
    lock_dir = path.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{path.name}.lock"
    with _ARTIFACT_LOCKS_GUARD:
        thread_lock = _ARTIFACT_LOCKS.setdefault(lock_path, threading.RLock())
    with thread_lock:
        depths = getattr(_ARTIFACT_LOCK_STATE, "depths", {})
        depth = int(depths.get(lock_path, 0))
        depths[lock_path] = depth + 1
        _ARTIFACT_LOCK_STATE.depths = depths
        if depth:
            try:
                yield
            finally:
                depths[lock_path] -= 1
            return
        acquired = False
        try:
            with open(lock_path, "a+b") as handle:
                _lock_handle(handle)
                acquired = True
                try:
                    yield
                finally:
                    if acquired:
                        _unlock_handle(handle)
        finally:
            depths.pop(lock_path, None)


class ReleasableArtifactLock:
    """Explicitly releasable form of :func:`artifact_lock`.

    Long-running fit pipelines sometimes need one short exclusive cache
    transaction followed by compute that must not retain the transaction lock.
    The owner still uses ``finally: release()``; ``release`` is idempotent so the
    inner cache stage may hand the lock back as soon as its pointer is durable.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._context: Any | None = None
        self._held = False

    def acquire(self) -> "ReleasableArtifactLock":
        if not self._held:
            context = artifact_lock(self._path)
            context.__enter__()
            self._context = context
            self._held = True
        return self

    def release(self) -> None:
        if self._held:
            self._held = False
            context = self._context
            self._context = None
            assert context is not None
            context.__exit__(None, None, None)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path``.

    Stages to a unique owner-only file in the same directory, fsyncs it,
    then publishes with ``os.replace()`` and syncs the parent directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name[:40]}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            tmp.unlink()


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file in bounded chunks, then publish it atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name[:40]}.", suffix=".tmp", dir=destination.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as output, open(source, "rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, destination)
        fsync_directory(destination.parent)
    finally:
        with suppress(FileNotFoundError):
            tmp.unlink()


def fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for prior directory-entry mutations."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(Path(path), flags)
    except OSError:
        # Windows and a few network filesystems do not permit directory opens.
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write ``payload`` as JSON to ``path``.

    Output ends with a trailing newline (matches the prior ``json.dump`` +
    ``f.write("\\n")`` convention used across the io layer).
    """
    text = json.dumps(payload, indent=indent) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))
