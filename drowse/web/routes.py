"""Mount the pre-built Svelte+Vite bundle from ``drowse/web/dist/``.

The bundle is a single-page app with hash routes.  A path the server does
not know answers ``index.html`` (an old ``/chat`` bookmark still lands on
the dashboard) unless it names a file: a path whose last segment is a
dotfile or has a static-file extension the bundle lacks
(``/apple-touch-icon.png``, ``/favicon.png/``) is a 404, never the shell
with a 200; a bipolar concept's dotted name (``/manifolds/ns/happy.sad``)
is still a route.  The shell is public: it holds
no data, and the API key is entered client-side and rides each API call
(only the API prefixes authenticate).  The entry chunks keep fixed names
(``assets/drowse.js``, ``assets/App.css``) while split chunks are
content-hashed, so every dashboard response says ``Cache-Control: no-cache``:
the browser revalidates each file by its ETag / Last-Modified on every load
(a 304 when nothing changed).  Without the header a browser may reuse a cached
``drowse.js`` on heuristic freshness and run an old dashboard against an
upgraded server.
"""
from __future__ import annotations

import re
from importlib import resources as _resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from drowse.errors import DrowseError

_REVALIDATE = {"Cache-Control": "no-cache"}

# A path segment that names a file (``favicon.png``, ``.env``), not a route:
# a dotfile or a static-file extension.  A bipolar concept's name holds a
# dot (``happy.sad``), so any other extension is still a route.
_FILE_NAME = re.compile(
    r"^\.|\.(?:png|jpe?g|gif|webp|avif|bmp|ico|svg|webmanifest|json|js|mjs|cjs|css|map"
    r"|txt|xml|html?|woff2?|ttf|otf|eot|wasm|pdf|zip|gz|mp4|webm|mp3|wav)$",
    re.IGNORECASE,
)


class _RevalidatingFiles(StaticFiles):
    """``StaticFiles`` whose responses always revalidate (``no-cache``)."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(_REVALIDATE)
        return response


class WebUINotBuilt(RuntimeError, DrowseError):
    """Raised when ``drowse serve`` runs against an empty dist dir.

    The wheel ships a pre-built bundle; this only fires in source-tree
    installs that haven't run ``cd webui && npm run build`` yet.  Pass
    ``--no-web`` to skip the dashboard mount when the bundle is missing
    on purpose.
    """

    def user_message(self) -> tuple[int, str]:
        return (500, "dashboard not built — run `cd webui && npm run build`")


def dist_path() -> Path:
    """Return the on-disk path to the bundled web UI assets.

    Resolved through ``importlib.resources`` so it works for both
    editable and wheel installs.  The traversable returned is always a
    ``Path`` because drowse itself ships as a regular filesystem
    package (not a zip).
    """
    files = _resources.files("drowse.web")
    dist = Path(str(files / "dist"))
    return dist


def register_web_routes(app: FastAPI) -> None:
    """Mount the SPA bundle at ``/`` with an SPA-fallback to index.html.

    Call it once per app.  A repeat call appends a second ``/assets``
    mount and duplicate ``/`` + catch-all routes; FastAPI matches in
    registration order, so the first set keeps serving and the extras sit
    shadowed — harmless but not rejected.  CLI default-on
    (``drowse serve``); ``--no-web`` opts out for production / proxied
    deployments.  Library callers using ``create_app`` directly
    default-off.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    dist = dist_path()
    if not dist.exists() or not (dist / "index.html").is_file():
        raise WebUINotBuilt(
            f"web UI bundle not found at {dist}. "
            f"Build it with: cd webui && npm ci && npm run build"
        )

    # The bundle's chunks, fonts and indexes live under ``assets/``, whose
    # mount answers a missing one with its own 404.
    static_dir = dist / "assets"
    if static_dir.is_dir():
        app.mount(
            "/assets",
            _RevalidatingFiles(directory=str(static_dir)),
            name="drowse-web-assets",
        )

    @app.get("/", include_in_schema=False)
    async def _root() -> FileResponse:
        return FileResponse(str(dist / "index.html"), headers=_REVALIDATE)

    # Catch-all for every other path. Won't shadow any earlier /api/*,
    # /v1/*, /drowse/v1/* routes because FastAPI evaluates in registration
    # order; create_app mounts this last.
    dist_root = dist.resolve()
    index_html = dist_root / "index.html"

    # Build an allowlist of top-level dist files at mount time
    # (favicon.ico, the marks, theme-preflight.js, the licences).
    # ``full_path`` from the request is then only ever used as a dict key —
    # never as a path component — so ``..`` traversal and absolute-path
    # injection are structurally impossible.
    top_level_files: dict[str, Path] = {
        p.name: p
        for p in dist_root.iterdir()
        if p.is_file() and p.name != "index.html"
    }

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        # An unknown API path is a 404, never the dashboard shell.
        if (
            full_path in {"api", "v1", "drowse"}
            or full_path.startswith(("api/", "v1/", "drowse/"))
        ):
            raise HTTPException(404, "not found")
        direct = top_level_files.get(full_path)
        if direct is not None:
            return FileResponse(str(direct), headers=_REVALIDATE)
        # A file the bundle lacks is a 404: the shell would answer a
        # browser's /apple-touch-icon.png or /manifest.webmanifest with
        # HTML and a 200.
        if _FILE_NAME.search(full_path.rstrip("/").rsplit("/", 1)[-1]):
            raise HTTPException(404, "not found")
        return FileResponse(str(index_html), headers=_REVALIDATE)
