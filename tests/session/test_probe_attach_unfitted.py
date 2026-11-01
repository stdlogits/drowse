"""Attaching a probe for an installed manifold that is not fitted for the model.

Regression: ``add_probe("default/emotions")`` on a model without an emotions
fit used to fall through to ``extract``, which slugged the selector into a
different one-node concept (``local/default_emotions``) and attached that.
Now the installed manifold is fitted in place (inside the attach's exclusive
section) and attached as itself; a selector that names nothing installed, or
a variant a fit cannot produce, is refused, and nothing else is authored.
"""
from __future__ import annotations

from typing import Any

import pytest

from drowse.base.paths import manifold_dir, manifolds_dir, tensor_filename
from drowse.errors import ManifoldNotRegisteredError, RequestError
from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.sessions import baseline_corpus

pytestmark = pytest.mark.usefixtures("isolated_home")

_LABELS = ("calm", "angry", "glad")


def _install_moods() -> None:
    create_discover_manifold_folder(
        "default", "moods", "three moods", fit_mode="pca",
        node_corpora={
            "calm": baseline_corpus("I feel calm and settled today."),
            "angry": baseline_corpus("I am furious about all of this!"),
            "glad": baseline_corpus("What a lovely, happy morning."),
        },
        hyperparams={"max_dim": 2},
    )


def _manifold_folders() -> set[str]:
    root = manifolds_dir()
    return {
        f"{ns.name}/{folder.name}"
        for ns in root.iterdir() if ns.is_dir()
        for folder in ns.iterdir() if (folder / "manifold.json").exists()
    }


def _fitted(session: Any) -> bool:
    return (manifold_dir("default", "moods") / tensor_filename(session.model_id)).exists()


@pytest.mark.parametrize("selector", ["default/moods", "moods"])
def test_an_unfitted_installed_manifold_is_fitted_and_attached_as_itself(
    tiny_session: Any, selector: str,
) -> None:
    _install_moods()
    before = _manifold_folders()
    assert not _fitted(tiny_session)

    assert tiny_session.add_probe(selector) == selector

    manifold = tiny_session.monitor.manifolds[selector]
    assert tuple(manifold.node_labels) == _LABELS
    assert _fitted(tiny_session)
    # Nothing else was authored: no local/default_moods, no local/moods.
    assert _manifold_folders() == before
    assert tiny_session.manifolds[selector] is manifold


def test_a_namespace_that_names_nothing_installed_is_refused(tiny_session: Any) -> None:
    before = _manifold_folders() if manifolds_dir().exists() else set()
    with pytest.raises(ManifoldNotRegisteredError, match="no manifold alice/moods is installed"):
        tiny_session.add_probe("alice/moods")
    after = _manifold_folders() if manifolds_dir().exists() else set()
    assert after == before
    assert "alice/moods" not in tiny_session.monitor.probe_names


def test_a_variant_a_probe_fit_cannot_produce_is_refused_before_fitting(
    tiny_session: Any,
) -> None:
    _install_moods()
    with pytest.raises(RequestError, match="--sae some-release"):
        tiny_session.add_probe("default/moods:sae-some-release")
    assert not _fitted(tiny_session)
    assert "default/moods:sae-some-release" not in tiny_session.monitor.probe_names


def test_the_probe_route_attaches_the_installed_manifold(
    tiny_session: Any, app_client: Any,
) -> None:
    _install_moods()
    client = app_client(tiny_session)
    url = "/drowse/v1/sessions/default/probes"
    response = client.post(url, json={"selector": "default/moods"})
    assert response.status_code == 201, response.text
    assert response.json()["name"] == "default/moods"
    assert "local/default_moods" not in _manifold_folders()

    missing = client.post(url, json={"selector": "alice/moods"})
    assert missing.status_code == 404, missing.text
    assert "no manifold alice/moods is installed" in missing.text
