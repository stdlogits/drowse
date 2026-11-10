"""Shared artifact primitives: the name grammar and content digests.

The cross-cutting infrastructure every persisted artifact family shares:

- ``NAME_REGEX`` — the artifact-name grammar.  Manifolds, selectors, templates,
  and the local lens/SAE source names all validate against this one pattern;
- ``LABEL_REGEX`` — the stricter manifold node-label grammar (no ``.``);
- ``is_sha256_digest`` / ``canonical_json`` — the one digest grammar and the
  one canonical-JSON encoding every content hash is taken over;
- ``hash_file`` — a file's sha256, for content identity (model fingerprints,
  a trained SAE's payload identity, provider lens bindings);
- ``PROFILE_FORMAT_VERSION`` — the current profile sidecar version written by
  :func:`drowse.geometry.profile.save_profile`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any



NAME_REGEX = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")

# Manifold node *labels* are stricter than ``NAME_REGEX``: they drop ``.``
# because the dot is the bipolar-pole separator (``deer.wolf``) and the
# steering grammar addresses a label via ``%label`` — a dotted label could
# neither be typed unambiguously nor resolved through the bare-name tier.  A
# templated manifold's node labels are its template's slugged values, so the
# template format shares this grammar.
LABEL_REGEX = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def is_sha256_digest(value: Any) -> bool:
    """True for a canonical lowercase 64-hex sha256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def canonical_json(obj: Any) -> bytes:
    """Stable canonical-JSON encoding (sorted keys, compact) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

# Current profile sidecar format version.  v6 is the slim schema
# (``format_version`` / ``drowse_version`` / ``method`` / ``provenance``; 6.1
# also wrote a ``tensor_sha256``, which the reader ignores), cut from fourteen
# keys.  No cache rides this version — the neutral-activation and alignment
# caches carry their own sidecar schemas and format versions in
# :mod:`drowse.store.neutral` and :mod:`drowse.store.alignment` — so the only
# reader affected is a user-saved ``Profile``, which gets a clear regenerate
# error.
PROFILE_FORMAT_VERSION = 6


def hash_file(path: Path) -> str:
    """Return hex sha256 of a file's contents."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
