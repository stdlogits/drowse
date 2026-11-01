"""The neuron-lens catalog binding over a monkeypatched Hub and tiny parquets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

pa = pytest.importorskip("pyarrow")

from drowse.base.integrity import hash_file  # noqa: E402
from drowse.naming import HeadRef, NeuronRef  # noqa: E402
from drowse.units import catalog as cat  # noqa: E402
from drowse.units.reads import HeadRead, NeuronRead  # noqa: E402

pytestmark = pytest.mark.usefixtures("isolated_home")

SLUG = "tiny-test"
N_LAYERS, D_FF, N_HEADS = 4, 64, 4


def _neuron_table(table: str, layer: int) -> Any:
    f = D_FF
    keys = {
        "layer": pa.array([layer] * 2 * f, pa.int16()),
        "neuron": pa.array(list(range(f)) * 2, pa.int16()),
        "pole": pa.array([1] * f + [-1] * f, pa.int8()),
    }
    rows = range(2 * f)
    if table == "labels":
        extra = {
            "fires": pa.array([f"L{layer} n{r % f} {'+' if r < f else '-'} fires" for r in rows]),
            "fires_conf": pa.array([0.5] * 2 * f, pa.float32()),
            "writes": pa.array([f"L{layer} n{r % f} writes" for r in rows]),
            "writes_conf": pa.array([0.25] * 2 * f, pa.float32()),
            "note": pa.array(["a note"] * 2 * f),
        }
    elif table == "acts":
        extra = {
            name: pa.array([0.001 * r + layer for r in rows], pa.float32())
            for name in cast(tuple[str, ...], cat.NEURON_COLUMNS["acts"])[3:]
        }
        extra["contexts"] = pa.array(["bulk context text"] * 2 * f)  # never fetched
    else:
        extra = {
            name: pa.array([False] * 2 * f) if name in ("silent", "dark") else pa.array([0.5] * 2 * f, pa.float32())
            for name in cast(tuple[str, ...], cat.NEURON_COLUMNS["legible"])[3:]
        }
        extra["cur_mode"] = pa.array([0.0] * 2 * f, pa.float32())  # never fetched
    return pa.table({**keys, **extra})


def _head_table(table: str) -> Any:
    n = N_LAYERS * N_HEADS
    keys = {
        "layer": pa.array([i // N_HEADS for i in range(n)], pa.int16()),
        "head": pa.array([i % N_HEADS for i in range(n)], pa.int16()),
        "kv_head": pa.array([(i % N_HEADS) // 2 for i in range(n)], pa.int16()),
    }
    if table == "labels":
        extra = {
            "fires": pa.array([f"head {i // N_HEADS}.{i % N_HEADS} fires" for i in range(n)]),
            "fires_conf": pa.array([0.5] * n, pa.float32()),
            "attends": pa.array(["the previous token"] * n),
            "attends_conf": pa.array([0.75] * n, pa.float32()),
            "writes": pa.array(["copies"] * n),
            "writes_conf": pa.array([0.5] * n, pa.float32()),
            "note": pa.array([""] * n),
        }
    elif table == "acts":
        extra = {
            name: pa.array([0.01 * i for i in range(n)], pa.float32())
            for name in cast(tuple[str, ...], cat.HEAD_COLUMNS["acts"])[3:]
        }
    else:
        extra = {
            name: pa.array([True] * n) if name in ("pattern", "dark") else pa.array([0.1] * n, pa.float32())
            for name in cast(tuple[str, ...], cat.HEAD_COLUMNS["legible"])[3:]
        }
    return pa.table({**keys, **extra})


def _parquet_paths() -> list[str]:
    paths = [
        f"{SLUG}/neurons/{table}/L{layer:02d}.parquet"
        for table in ("labels", "acts", "legible") for layer in range(N_LAYERS)
    ]
    return paths + [f"{SLUG}/heads/{table}.parquet" for table in ("labels", "acts", "legible")]


#: The catalog's token counts upstream (a torch pickle of plain tensors).
COUNTS_PATH = f"{SLUG}/token_counts.pt"
#: The fake counts: every id seen 10 times but :data:`RARE_IDS` (twice): in the
#: tiny byte tokenizer, "{", "|" and "~".
COUNTS_VOCAB = 512
RARE_IDS = (95, 96, 98)


def _fake_counts() -> Any:
    import torch

    counts = torch.full((COUNTS_VOCAB,), 10, dtype=torch.int64)
    counts[list(RARE_IDS)] = 2
    return counts


def _remote_paths() -> list[str]:
    return [*_parquet_paths(), COUNTS_PATH]


#: Files the dataset holds that a binding never copies: the bulk tables, the
#: causal sample, media, documents, another model's folder, and paths of the
#: wrong depth for their kind.
_SKIPPED_PATHS = (
    f"{SLUG}/neurons/reads/L00.parquet",
    f"{SLUG}/neurons/drivers/L01.parquet",
    f"{SLUG}/heads/reads.parquet",
    f"{SLUG}/heads/drivers.parquet",
    f"{SLUG}/causal/sample.parquet",
    f"{SLUG}/media/L00/n0.png",
    f"{SLUG}/docs.parquet",
    f"{SLUG}/neurons/labels/README.md",
    f"{SLUG}/neurons/labels.parquet",
    f"{SLUG}/heads/labels/L00.parquet",
    "other-model/neurons/labels/L00.parquet",
    "other-model/heads/labels.parquet",
    "README.md",
)


def _upstream_sha(path: str) -> str:
    import hashlib

    return hashlib.sha256(path.encode()).hexdigest()


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A fake Hub: the dataset's sibling listing (the real path filter runs
    over it) and projected reads, with the calls recorded."""
    from types import SimpleNamespace

    import huggingface_hub

    calls: dict[str, Any] = {"listings": 0, "reads": [], "downloads": []}

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> Any:
            assert (repo_id, revision, files_metadata) == (cat.CATALOG_REPO, cat.CATALOG_REVISION, True)
            calls["listings"] += 1
            siblings = [
                SimpleNamespace(
                    rfilename=path, size=1000 + i,
                    lfs=SimpleNamespace(sha256=_upstream_sha(path), size=1000 + i),
                )
                for i, path in enumerate(sorted([*_remote_paths(), *_SKIPPED_PATHS]))
            ]
            return SimpleNamespace(siblings=siblings)

    def read_remote(path: str, revision: str, columns: Any) -> Any:
        assert revision == cat.CATALOG_REVISION
        calls["reads"].append((path, None if columns is None else tuple(columns)))
        parts = path.split("/")
        table = (
            _neuron_table(parts[2], int(parts[3][1:3])) if parts[1] == "neurons"
            else _head_table(parts[2].split(".")[0])
        )
        return table if columns is None else table.select(list(columns))

    def download(path: str, revision: str) -> Path:
        import torch

        assert (path, revision) == (COUNTS_PATH, cat.CATALOG_REVISION)
        calls["downloads"].append(path)
        target = tmp_path / f"download-{len(calls['downloads'])}" / "token_counts.pt"
        target.parent.mkdir()
        torch.save({
            "counts": _fake_counts(), "by_slice": {"dclm": _fake_counts()}, "tokenizer": "test",
        }, target)
        return target

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(cat, "_read_remote_table", read_remote)
    monkeypatch.setattr(cat, "_download_remote", download)
    return calls


@pytest.fixture
def catalogued(monkeypatch: pytest.MonkeyPatch) -> str:
    model_id = "test-org/Tiny-Catalogued"
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, model_id.lower(): cat.CatalogModel(model_id, SLUG),
    })
    return model_id


def test_fetch_binds_labels_and_projected_tables(hub: dict[str, Any], catalogued: str) -> None:
    progress: list[str] = []
    record = cat.fetch_catalog(catalogued, on_progress=progress.append)
    root = cat.catalog_dir(catalogued)
    assert record["repo"] == cat.CATALOG_REPO and record["revision"] == cat.CATALOG_REVISION
    assert record["slug"] == SLUG and record["credit"] == "Labels by Claude Haiku 5.5"
    assert len(record["files"]) == len(_remote_paths())
    on_disk = cat._read_binding(root)
    assert on_disk == record
    entry = record["files"]["neurons/acts/L02.parquet"]
    assert entry["upstream"] == f"{SLUG}/neurons/acts/L02.parquet"
    assert entry["sha256"] == hash_file(root / "neurons/acts/L02.parquet")
    assert entry["rows"] == 2 * D_FF
    # The listing's path filter keeps exactly this model's label, acts and
    # legible tables (4-part neuron paths, 3-part head paths) and records
    # each one's upstream LFS digest; nothing else is read.
    assert sorted(item["upstream"] for item in record["files"].values()) == sorted(_remote_paths())
    assert all(
        item["upstream_sha256"] == _upstream_sha(item["upstream"]) for item in record["files"].values()
    )
    assert sorted(path for path, _ in hub["reads"]) == sorted(_parquet_paths())
    assert hub["downloads"] == [COUNTS_PATH]
    entry = record["files"][cat.TOKEN_COUNTS_FILE]
    assert entry["upstream"] == COUNTS_PATH and entry["rows"] == COUNTS_VOCAB
    assert entry["sha256"] == hash_file(root / cat.TOKEN_COUNTS_FILE)
    # Labels whole; acts and legible projected (the bulk columns never asked for).
    asked = dict(hub["reads"])
    assert asked[f"{SLUG}/neurons/labels/L00.parquet"] is None
    acts_columns = asked[f"{SLUG}/neurons/acts/L00.parquet"]
    assert acts_columns == cat.NEURON_COLUMNS["acts"] and "contexts" not in acts_columns
    import pyarrow.parquet as pq

    assert "contexts" not in pq.read_table(root / "neurons/acts/L00.parquet").column_names
    assert progress[-1].startswith("bound ")
    # Current: a second fetch lists nothing; force refetches.
    cat.fetch_catalog(catalogued)
    assert hub["listings"] == 1
    cat.fetch_catalog(catalogued, force=True)
    assert hub["listings"] == 2
    status = cat.catalog_status(catalogued)
    assert status["bound"] and status["available"] and status["reason"] is None
    assert status["binding"]["files"] == len(_remote_paths())


def test_catalog_reads_rows_labels_and_links(hub: dict[str, Any], catalogued: str) -> None:
    cat.fetch_catalog(catalogued)
    catalog = cat.UnitCatalog.open(catalogued)
    assert catalog is not None
    plus = catalog.neuron_label(2, 7, 1)
    minus = catalog.neuron_label(2, 7, -1)
    assert plus is not None and minus is not None
    assert plus["fires"] == "L2 n7 + fires" and minus["fires"] == "L2 n7 - fires"
    assert plus["writes_conf"] == pytest.approx(0.25)
    assert plus["credit"] == "Labels by Claude Haiku 5.5"
    assert plus["url"] == f"https://neuron-lens.com/{SLUG}/L2/n7"
    assert plus["legible"] is not None and plus["legible"]["dark"] is False
    acts = catalog.neuron_acts(2, 7, -1)
    assert acts is not None and acts["rate"] == pytest.approx(0.001 * (D_FF + 7) + 2, rel=1e-5)
    assert set(acts) == set(cast(tuple[str, ...], cat.NEURON_COLUMNS["acts"])[3:])
    head = catalog.head_label(3, 1)
    assert head is not None and head["fires"] == "head 3.1 fires" and head["attends"] == "the previous token"
    assert head["url"] == f"https://neuron-lens.com/{SLUG}/L3/h1"
    head_acts = catalog.head_acts(3, 1)
    assert head_acts is not None and head_acts["rate"] == pytest.approx(0.01 * (3 * N_HEADS + 1), rel=1e-5)
    assert catalog.neuron_label(2, D_FF, 1) is None and catalog.head_label(N_LAYERS, 0) is None


def test_neuron_lens_urls_match_the_explorer_routes() -> None:
    assert cat.unit_url("qwen3-8b", NeuronRef(layer=8, index=344)) == "https://neuron-lens.com/qwen3-8b/L8/n344"
    assert cat.unit_url("gemma-4-12b", HeadRef(layer=35, index=14)) == "https://neuron-lens.com/gemma-4-12b/L35/h14"
    assert cat.catalog_model("qwen/QWEN3-8b") == cat.CatalogModel("Qwen/Qwen3-8B", "qwen3-8b")
    assert cat.catalog_model("google/gemma-4-12b-it") is None


def test_uncatalogued_models_are_a_clear_404(hub: dict[str, Any]) -> None:
    with pytest.raises(cat.NoUnitCatalogError) as caught:
        cat.fetch_catalog("HuggingFaceTB/SmolLM2-360M-Instruct")
    status, message = caught.value.user_message()
    assert status == 404 and "Qwen/Qwen3-8B" in message and "google/gemma-4-12B" in message
    assert hub["listings"] == 0
    status_row = cat.catalog_status("HuggingFaceTB/SmolLM2-360M-Instruct")
    assert not status_row["available"] and not status_row["bound"]


def test_a_binding_of_another_revision_is_not_read(hub: dict[str, Any], catalogued: str) -> None:
    cat.fetch_catalog(catalogued)
    root = cat.catalog_dir(catalogued)
    import json

    record = json.loads((root / "catalog.json").read_text())
    (root / "catalog.json").write_text(json.dumps({**record, "revision": "0" * 40}))
    assert cat.UnitCatalog.open(catalogued) is None
    status = cat.catalog_status(catalogued)
    assert not status["bound"] and "-f" in status["reason"]
    assert cat.remove_catalog(catalogued) and not Path(root).exists()
    assert "not fetched" in cat.catalog_status(catalogued)["reason"]


def test_missing_pyarrow_is_a_503(monkeypatch: pytest.MonkeyPatch, catalogued: str) -> None:
    import builtins

    real = builtins.__import__

    def no_arrow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_arrow)
    with pytest.raises(cat.UnitCatalogDependencyError) as caught:
        cat.fetch_catalog(catalogued)
    assert caught.value.user_message()[0] == 503


def test_session_unit_views_carry_the_catalog(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, make_tiny_session: Any,
) -> None:
    session = make_tiny_session()
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, session.model_id.lower(): cat.CatalogModel(session.model_id, SLUG),
    })
    before = cast(NeuronRead, session.units.read("neuron/L1n3")).to_dict()
    assert before["catalog"]["bound"] is False and "not fetched" in before["catalog"]["reason"]
    record = session.units.fetch_catalog()
    assert record["slug"] == SLUG and session.units.catalog_status()["bound"]
    read = cast(NeuronRead, session.units.read("neuron/L1n3")).to_dict()
    assert read["catalog"] == {
        "bound": True, "slug": SLUG, "url": f"https://neuron-lens.com/{SLUG}/L1/n3",
        "credit": "Labels by Claude Haiku 5.5", "reason": None,
    }
    minus = read["poles"][1]
    assert minus["label"]["fires"] == "L1 n3 - fires"
    assert minus["firing"]["rate"] == pytest.approx(0.001 * (D_FF + 3) + 1, rel=1e-4)
    assert minus["firing"]["catalog"]["act_top"] is not None
    head = cast(HeadRead, session.units.read("head/L2h1")).to_dict()
    assert head["label"]["attends"] == "the previous token"
    assert head["firing"]["catalog"]["induction"] is not None
    stats = session.units.stats("neuron/L1n3")
    assert stats["poles"][0]["label"]["url"].endswith("/L1/n3")
    top = session.units.top("a", layers=[1], k=2)
    assert all(hit.label is not None and hit.label["fires"].startswith("L1 ") for hit in top.hits)


def test_a_catalog_bound_by_another_process_reaches_a_running_session(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, make_tiny_session: Any,
) -> None:
    """Regression: the session's binding cached its first open, a "no
    catalog" included, so a ``drowse unit fetch`` from another terminal never
    reached a running server, and its unit views said ``bound: false`` with
    no reason.  Every view now rechecks ``catalog.json`` on disk."""
    session = make_tiny_session()
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, session.model_id.lower(): cat.CatalogModel(session.model_id, SLUG),
    })
    binding = session.units.reads.catalog
    before = cast(NeuronRead, session.units.read("neuron/L1n3")).to_dict()
    assert before["catalog"]["bound"] is False and "not fetched" in before["catalog"]["reason"]
    assert binding.get() is None
    # Another process binds it: the module function, not the session's binding.
    cat.fetch_catalog(session.model_id)
    read = cast(NeuronRead, session.units.read("neuron/L1n3")).to_dict()
    assert read["catalog"] == {
        "bound": True, "slug": SLUG, "url": f"https://neuron-lens.com/{SLUG}/L1/n3",
        "credit": "Labels by Claude Haiku 5.5", "reason": None,
    }
    assert read["poles"][0]["label"]["fires"] == "L1 n3 + fires"
    assert read["poles"][0]["firing"]["rate"] is not None
    opened = binding.get()
    assert opened is not None and binding.get() is opened  # unchanged on disk: no reopen
    # Refetched elsewhere: the new binding on disk is reopened.
    cat.fetch_catalog(session.model_id, force=True)
    reopened = binding.get()
    assert reopened is not None and reopened is not opened
    # Removed elsewhere: the views say so, with the reason.
    assert cat.remove_catalog(session.model_id)
    stats = session.units.stats("neuron/L1n3")
    assert stats["catalog"]["bound"] is False and "not fetched" in stats["catalog"]["reason"]
    assert stats["poles"][0]["label"] is None
    head = cast(HeadRead, session.units.read("head/L2h1")).to_dict()
    assert head["catalog"]["bound"] is False and head["catalog"]["reason"]
    assert head["firing"]["catalog_absent"] == head["catalog"]["reason"]


def test_a_binding_view_never_contradicts_itself(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, catalogued: str,
) -> None:
    """A status that says bound while the catalog does not open (a binding
    replaced mid-read) still gives a reason."""
    cat.fetch_catalog(catalogued)
    binding = cat.CatalogBinding(catalogued)
    monkeypatch.setattr(cat.UnitCatalog, "open", classmethod(lambda cls, model_id: None))
    catalog, status = binding.view()
    assert catalog is None and status["bound"] is False and status["reason"]


def test_unit_activity_peaks_carry_the_label_of_the_pole_that_peaked(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, make_tiny_session: Any,
) -> None:
    """The Units room lists a conversation's most active units with their
    labels and filters to the labelled ones, so each activity peak carries
    the bound catalog's label of the pole it peaked on (a head's label)."""
    from drowse.generation.sampling import SamplingConfig

    session = make_tiny_session()
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, session.model_id.lower(): cat.CatalogModel(session.model_id, SLUG),
    })
    session.generate("hello", sampling=SamplingConfig(max_tokens=5, seed=0))
    node_id = session.tree.active_node_id
    unbound = session.unit_activity(node_id, top=24)
    assert unbound["top"] and all(peak["label"] is None for peak in unbound["top"])
    assert unbound["scale"]["source"] == "neutral"
    session.units.fetch_catalog()
    bound = session.unit_activity(node_id, top=24)
    # Bound, the values read over the catalog's act_top and say so.
    assert bound["scale"] == {
        "source": "catalog", "detail": f"neuron-lens catalog {cat.CATALOG_REPO}@{cat.CATALOG_REVISION[:12]}",
    }
    peaks = bound["top"]
    assert peaks
    for peak in peaks:
        label = peak["label"]
        assert label is not None and label["credit"] == "Labels by Claude Haiku 5.5"
        if peak["kind"] == "neuron":
            pole = "+" if peak["peak"] >= 0 else "-"
            assert label["fires"] == f"L{peak['layer']} n{peak['index']} {pole} fires"
            assert label["url"].endswith(f"/L{peak['layer']}/n{peak['index']}")
        else:
            assert label["attends"] == "the previous token"
            assert label["url"].endswith(f"/L{peak['layer']}/h{peak['index']}")


def test_labels_read_every_pole_from_the_catalog(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, make_tiny_session: Any,
) -> None:
    """``units.labels`` (the token anatomy's rows): both poles of a neuron,
    a head once, each unit once in the order asked; unbound, no rows and the
    reason."""
    session = make_tiny_session()
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, session.model_id.lower(): cat.CatalogModel(session.model_id, SLUG),
    })
    unbound = session.units.labels(["neuron/L1n3"])
    assert unbound["labels"] == [] and unbound["catalog"]["bound"] is False
    assert "not fetched" in unbound["catalog"]["reason"]
    session.units.fetch_catalog()
    out = session.units.labels(["head/L2h1", "neuron/L1n3", "neuron/L1n3"])
    assert out["catalog"] == {
        "bound": True, "slug": SLUG, "credit": "Labels by Claude Haiku 5.5", "reason": None,
    }
    assert [(row["unit"], row["pole"]) for row in out["labels"]] == [
        ("head/L2h1", None), ("neuron/L1n3", "+"), ("neuron/L1n3", "-"),
    ]
    head, plus, minus = (row["label"] for row in out["labels"])
    assert head["attends"] == "the previous token"
    assert plus["fires"] == "L1 n3 + fires" and minus["fires"] == "L1 n3 - fires"
    assert plus["url"] == f"https://neuron-lens.com/{SLUG}/L1/n3"


def test_a_bound_catalog_sets_the_reading_scale_and_the_token_mask(
    hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch, make_tiny_session: Any,
) -> None:
    """Decision 1 and 2 (Qwen3-8B: the neutral act_top sat a median 12x below
    the catalog's, and tokens neuron-lens saw under 5 times in 88M topped the
    weighted S half).  With a catalog bound, every reading divides by the
    catalog's act_top (soft-floored by the layer's catalog median) and says
    so, and the S mask keeps the catalog's tokens seen 5+ times."""
    import torch

    from drowse.generation.sampling import SamplingConfig
    from drowse.units.stats import SCALE_FLOOR_SHARE, typical_scale

    session = make_tiny_session()
    monkeypatch.setattr(cat, "CATALOG_MODELS", {
        **cat.CATALOG_MODELS, session.model_id.lower(): cat.CatalogModel(session.model_id, SLUG),
    })
    session.generate("hello", sampling=SamplingConfig(max_tokens=4, seed=0))
    node_id = session.tree.active_node_id
    neutral_mask = session.lens.token_mask_info()
    assert neutral_mask["source"] == "tokenizer" and "drowse lens counts" in neutral_mask["detail"]
    neutral_kept = session.lens.token_embeddings().n_kept
    neutral = session.unit_activity(node_id, top=4, units=["neuron/L1n3", "head/L2h1"])
    session.units.fetch_catalog()

    scales = session.units.runtime.scales()
    assert scales.source == "catalog"
    # The fake catalog's act_top: 0.001·row + layer (rows: + poles, then −).
    rows = torch.arange(2 * D_FF, dtype=torch.float32).reshape(2, D_FF)
    top = 0.001 * rows + 1
    torch.testing.assert_close(scales.layer(1).neuron_top, top)
    m = typical_scale(top)
    torch.testing.assert_close(scales.layer(1).neuron, torch.sqrt(top ** 2 + (SCALE_FLOOR_SHARE * m) ** 2))
    head_top = 0.01 * torch.arange(N_LAYERS * N_HEADS, dtype=torch.float32).reshape(N_LAYERS, N_HEADS)
    torch.testing.assert_close(scales.layer(2).head_top, head_top[2])

    bound = session.unit_activity(node_id, top=4, units=["neuron/L1n3", "head/L2h1"])
    assert bound["scale"]["source"] == "catalog"
    raw = session.unit_activity(node_id, top=4, units=["neuron/L1n3"])["traces"]["neuron/L1n3"]
    # Same activation, new divisor: trace × neutral scale == trace × catalog scale.
    stats = session.units.runtime.stats()
    neutral_scale = stats.reading_scales.layer(1).neuron[:, 3]
    catalog_scale = scales.layer(1).neuron[:, 3]
    for before, after in zip(neutral["traces"]["neuron/L1n3"], raw, strict=True):
        pole = 0 if before >= 0 else 1
        assert before * float(neutral_scale[pole]) == pytest.approx(
            after * float(catalog_scale[pole]), rel=1e-4, abs=1e-6,
        )
    # The unit view and a probe reading name the source too.
    pole = cast(NeuronRead, session.units.read("neuron/L1n3")).to_dict()["poles"][0]
    assert pole["firing"]["scale"]["source"] == "catalog"
    assert pole["firing"]["scale"]["top"] == pytest.approx(float(top[0, 3]), rel=1e-5)
    assert pole["firing"]["scale"]["value"] == pytest.approx(float(catalog_scale[0]), rel=1e-5)
    session.add_probe("neuron/L1n3")
    replay = session.units.token_readout(node_id, 1)["measurements"]
    reading = replay["instruments"]["units"]["readings"]["neuron/L1n3"]
    assert reading["meta"]["scale"] == "catalog"

    # The S mask: the catalog's counts, the ids seen twice dropped.
    info = session.lens.token_mask_info()
    assert info["source"] == "catalog" and "kept at 5+" in info["detail"]
    emb = session.lens.token_embeddings()
    assert emb.mask_source == "catalog" and emb.n_kept < neutral_kept
    keep = emb.keep.cpu()
    assert not any(bool(keep[i]) for i in RARE_IDS)
    assert emb.n_kept == neutral_kept - len(RARE_IDS)
    assert session.lens.token_mask_info()["kept"] == emb.n_kept
    # Removed: back to the neutral scale and the tokenizer mask.
    assert session.units.reads.catalog.remove()
    assert session.units.runtime.scales().source == "neutral"
    assert session.lens.token_embeddings().mask_source == "tokenizer"
