"""A fit made under an older fit policy (or for an edited corpus) is a miss:
the probe resolver refits it in place instead of failing the attach."""
from __future__ import annotations

import json
from typing import Any

import pytest

from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.sessions import baseline_corpus


@pytest.mark.usefixtures("isolated_home")
def test_a_probe_with_an_older_fit_policy_refits_in_place(make_tiny_session: Any) -> None:
    from drowse.base.paths import manifold_dir, tensor_filename
    from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION

    create_discover_manifold_folder(
        "local", "mood", "test axis", fit_mode="pca",
        node_corpora={
            "calm": baseline_corpus("I feel calm and settled today."),
            "angry": baseline_corpus("I am furious about all of this!"),
        },
        hyperparams={"max_dim": 1},
    )
    session = make_tiny_session()
    session.fit(manifold_dir("local", "mood"))
    sidecar = (manifold_dir("local", "mood") / tensor_filename(session.model_id)).with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    payload["fit_policy_version"] = MANIFOLD_FIT_POLICY_VERSION - 1
    sidecar.write_text(json.dumps(payload))

    fresh = make_tiny_session()
    assert fresh.add_probe("local/mood") == "local/mood"
    assert json.loads(sidecar.read_text())["fit_policy_version"] == MANIFOLD_FIT_POLICY_VERSION
