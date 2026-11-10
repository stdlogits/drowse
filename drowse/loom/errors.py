"""Loom error types and the reader field helpers they back.

Every persisted loom shape (tree, node, recipe, cast member) is read the same
way: every field the reader uses is required, and any other key is ignored.
The helpers below are that rule; they raise :class:`LoomTreeError` so a
document missing a field fails with a 400-class error instead of a bare
``KeyError``.
"""

from __future__ import annotations

from typing import Any

from drowse.errors import DrowseError, RequestError


class LoomTreeError(DrowseError):
    """Base class for tree-mutation errors."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class UnknownNodeError(KeyError, LoomTreeError):
    """Raised when a node id is not in the tree."""

    def user_message(self) -> tuple[int, str]:
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class InvalidNodeOperationError(ValueError, LoomTreeError):
    """Raised when an op is semantically invalid (e.g. deleting an ancestor of active)."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class RecipeOverrideError(RequestError):
    """Raised when a ``recipe_override`` string is not a mode or a valid partial
    recipe (unknown field, duplicate, malformed or out-of-range value)."""


class MutationDuringGenerationError(RuntimeError, LoomTreeError):
    """Raised when a tree mutation conflicts with an in-flight generation.

    A generation reserves the subtree rooted at the parent of the node it
    streams into (:meth:`LoomTree.reserve`).  Decoration ops, branches and
    new turns are always free; edits, appends and deletes touching that
    reservation, and a reset, refuse with this error (HTTP 409).
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


def require_fields(data: dict[str, Any], required: frozenset[str], label: str) -> None:
    """Reject a document missing a required key (other keys are ignored)."""
    missing = sorted(required - data.keys())
    if missing:
        raise LoomTreeError(f"{label} is missing required fields: {', '.join(missing)}")


def is_number(value: Any) -> bool:
    """A JSON number: ``int``/``float`` but never ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "InvalidNodeOperationError",
    "LoomTreeError",
    "MutationDuringGenerationError",
    "RecipeOverrideError",
    "UnknownNodeError",
]
