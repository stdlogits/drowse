from __future__ import annotations

import json
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.geometry.domains import CustomDomain
from drowse.geometry.subspace import LayerSubspace
from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION, Manifold
from drowse.store.manifold_authoring import (
    create_discover_manifold_folder,
    create_manifold_folder,
    create_manifold_from_template,
    iter_manifold_folders,
)
from drowse.store.manifold_tensors import save_manifold
from drowse.store.manifold_folder import ManifoldFolder
from drowse.store.manifold_schema import ManifoldFormatError
from drowse.base.paths import manifold_dir, drowse_home, tensor_filename
from drowse.store.archive import (
    DROWSE_ARCHIVE_FORMAT_VERSION,
    DrowseArchiveError,
    _validate_member_path,
    _validate_zip_infos,
    export_drowse_archive,
)
from drowse.store.install import install_manifold
from drowse.store.templates import (
    TemplateFolder,
    create_template_folder,
    template_dir,
)


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))


def _discover(name: str = "demo") -> Path:
    return create_discover_manifold_folder(
        "local",
        name,
        "demo manifold",
        fit_mode="pca",
        node_corpora={"calm": ["calm"], "alert": ["alert"]},
        hyperparams={"max_dim": 1},
    )


def _template_manifold() -> tuple[Path, Path]:
    template = create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Monday", "Tuesday"],
        contexts=[{
            "turns": [{"role": "user", "content": "what day is it?"}],
            "assistant": "today is [DAY]",
        }],
    )
    manifold = create_manifold_from_template(
        "local",
        "days",
        "days",
        template_ref="local/weekday",
        fit_mode="pca",
    )
    assert template.path is not None
    return manifold, template.path


def _fit(folder: Path, *, finite: bool = True) -> Path:
    mf = ManifoldFolder.load(folder)
    basis = torch.tensor([[1.0, 0.0]])
    if not finite:
        basis[0, 0] = float("nan")
    node_coords = torch.tensor([[-1.0], [1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2), basis=basis, read=basis.clone(),
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    path = folder / tensor_filename("test/model")
    save_manifold(
        manifold,
        path,
        {
            "method": "manifold_discover_pca",
            "fit_mode": "pca",
            "hyperparams": dict(mf.hyperparams),
            "nodes_sha256": mf.nodes_sha256(),
            "model_fingerprint": "test-model",
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
        },
    )
    return path


def _archive_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _write_entries(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _rewrite_manifest(
    source: Path, target: Path, mutate: Any,
) -> None:
    entries = _archive_entries(source)
    manifest = json.loads(entries["pack.json"])
    mutate(manifest, entries)
    entries["pack.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    _write_entries(target, entries)


def _replace_manifested_json(
    manifest: dict[str, Any],
    entries: dict[str, bytes],
    path: str,
    payload: dict[str, Any] | list[Any],
) -> None:
    entries[path] = (json.dumps(payload, sort_keys=True) + "\n").encode()


def test_round_trip_preserves_manifold_and_exact_manifest(tmp_path: Path) -> None:
    folder = _discover()
    output = tmp_path / "demo.drowse"

    assert export_drowse_archive("local", "demo", output=output) == output
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("pack.json"))
    assert manifest["format_version"] == DROWSE_ARCHIVE_FORMAT_VERSION
    assert manifest["kind"] == "drowse-manifold"
    assert manifest["primary"] == "manifolds/local/demo"
    assert manifest["template"] is None
    assert "files" not in manifest
    assert "manifolds/local/demo/manifold.json" in names

    shutil.rmtree(folder)
    installed = install_manifold(output)
    assert installed == manifold_dir("local", "demo")
    assert ManifoldFolder.load(installed).node_labels == ["calm", "alert"]


def test_monopolar_discover_fit_exports_and_installs(tmp_path: Path) -> None:
    folder = create_discover_manifold_folder(
        "local",
        "monopolar",
        "monopolar manifold",
        fit_mode="pca",
        node_corpora={"calm": ["calm"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
    )
    mf = ManifoldFolder.load(folder)
    node_coords = torch.tensor([[1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2),
                basis=torch.tensor([[1.0, 0.0]]),
                read=torch.tensor([[1.0, 0.0]]),
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    tensor = folder / tensor_filename("test/model")
    save_manifold(
        manifold,
        tensor,
        {
            "method": "manifold_monopolar",
            "fit_mode": mf.fit_mode,
            "hyperparams": dict(mf.hyperparams),
            "diagnostics": {},
            "nodes_sha256": mf.nodes_sha256(),
            "model_fingerprint": "test-model",
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
        },
    )

    output = tmp_path / "monopolar.drowse"
    export_drowse_archive("local", "monopolar", output=output)
    shutil.rmtree(folder)
    installed = install_manifold(output)

    sidecar = ManifoldFolder.load(installed).sidecar(tensor.stem)
    assert sidecar.method == "manifold_monopolar"
    assert sidecar.fit_mode == "pca"
    assert sidecar.hyperparams == {"max_dim": 1, "var_threshold": 0.7}
    assert sidecar.diagnostics == {}


def test_template_closure_installs_transactionally(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("pack.json"))
        names = set(archive.namelist())
    assert manifest["template"] == "templates/local/weekday"
    assert "templates/local/weekday/template.json" in names

    shutil.rmtree(folder)
    shutil.rmtree(template)
    install_manifold(output)
    assert ManifoldFolder.load(manifold_dir("local", "days")).template_ref == (
        "local/weekday"
    )
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Monday", "Tuesday",
    )


def test_export_rejects_unqualified_template_ref(tmp_path: Path) -> None:
    folder, _template = _template_manifold()
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["template_ref"] = "weekday"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "days.drowse"
    with pytest.raises(DrowseArchiveError, match="namespace-qualified"):
        export_drowse_archive("local", "days", output=output)


def test_different_existing_template_blocks_whole_install(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )

    with pytest.raises(RuntimeError, match="different referenced template"):
        install_manifold(output)
    assert not manifold_dir("local", "days").exists()
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Friday", "Saturday",
    )


def test_force_replaces_template_and_manifold_together(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )

    install_manifold(output, force=True)
    assert ManifoldFolder.load(manifold_dir("local", "days")).name == "days"
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Monday", "Tuesday",
    )


def test_force_refuses_to_replace_template_shared_by_another_manifold(
    tmp_path: Path,
) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )
    create_manifold_from_template(
        "local",
        "other",
        "other days",
        template_ref="local/weekday",
        fit_mode="pca",
    )

    with pytest.raises(RuntimeError, match="cannot replace shared template"):
        install_manifold(output, force=True)

    assert not manifold_dir("local", "days").exists()
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Friday", "Saturday",
    )


def test_as_rewrites_manifold_and_fitted_sidecar_identity(tmp_path: Path) -> None:
    folder = _discover()
    _fit(folder)
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)

    installed = install_manifold(output, as_="shared/renamed")
    loaded = ManifoldFolder.load(installed)
    assert installed == manifold_dir("shared", "renamed")
    assert loaded.name == "renamed"
    with open(installed / tensor_filename("test/model").replace(
        ".safetensors", ".json",
    )) as handle:
        sidecar = json.load(handle)
    assert sidecar["name"] == "renamed"


@pytest.mark.parametrize("namespace", ["jlens", "sae"])
def test_install_rejects_reserved_destination_namespace(
    tmp_path: Path, namespace: str,
) -> None:
    folder = _discover()
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)

    with pytest.raises(ValueError, match="reserved"):
        install_manifold(output, as_=f"{namespace}/renamed")


def test_install_rejects_reserved_source_namespace(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        old = "manifolds/local/demo"
        new = "manifolds/jlens/demo"
        manifest["primary"] = new
        for path in [name for name in entries if name.startswith(old + "/")]:
            entries[new + path[len(old):]] = entries.pop(path)

    _rewrite_manifest(valid, bad, mutate)

    with pytest.raises(DrowseArchiveError, match="reserved"):
        install_manifold(bad)


def test_fit_transaction_ownership_never_travels(tmp_path: Path) -> None:
    folder = _discover()
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_id"] = "publisher-owned"
    manifest["fit_epochs"] = {"*:all": 91}
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    with zipfile.ZipFile(output) as archive:
        archived = json.loads(
            archive.read("manifolds/local/demo/manifold.json")
        )
    assert "artifact_id" not in archived
    assert "fit_epochs" not in archived

    # An archive authored elsewhere (the browser allows the keys) is reset on
    # install as well.
    foreign = tmp_path / "foreign.drowse"

    def carry_ownership(pack: dict[str, Any], entries: dict[str, bytes]) -> None:
        _replace_manifested_json(
            pack, entries, "manifolds/local/demo/manifold.json",
            {**archived, "artifact_id": "publisher-owned", "fit_epochs": {"*:all": 91}},
        )

    _rewrite_manifest(output, foreign, carry_ownership)
    shutil.rmtree(folder)
    installed = install_manifold(foreign)
    local_manifest = json.loads((installed / "manifold.json").read_text())
    assert "artifact_id" not in local_manifest
    assert "fit_epochs" not in local_manifest


def test_export_requires_drowse_archive_suffix(tmp_path: Path) -> None:
    _discover()
    with pytest.raises(DrowseArchiveError, match="must end in .drowse"):
        export_drowse_archive("local", "demo", output=tmp_path / "demo.zip")


def test_export_refuses_destination_inside_manifold_closure() -> None:
    folder = _discover()
    destination = folder / "nested.drowse"
    with pytest.raises(DrowseArchiveError, match="outside its manifold closure"):
        export_drowse_archive("local", "demo", output=destination)
    assert not destination.exists()


def test_export_refuses_destination_inside_template_closure() -> None:
    _folder, template = _template_manifold()
    destination = template / "nested.drowse"
    with pytest.raises(DrowseArchiveError, match="outside its template closure"):
        export_drowse_archive("local", "days", output=destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "hyperparams",
    [{"k_nn": 3}, {"max_dim": True}, {"var_threshold": float("nan")}],
)
def test_discover_hyperparameters_are_mode_validated(
    hyperparams: dict[str, object],
) -> None:
    with pytest.raises(ManifoldFormatError):
        create_discover_manifold_folder(
            "local",
            "invalid",
            "invalid",
            fit_mode="pca",
            node_corpora={"calm": ["calm"], "alert": ["alert"]},
            hyperparams=hyperparams,
        )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "manifolds\\local\\demo",
        "manifolds//demo",
        "manifolds/./demo",
        "manifolds/../demo",
        "bad\x00path",
    ],
)
def test_rejects_unsafe_member_paths(name: str) -> None:
    with pytest.raises(DrowseArchiveError):
        _validate_member_path(name)


def test_rejects_duplicate_and_casefold_colliding_paths() -> None:
    duplicate = [zipfile.ZipInfo("pack.json"), zipfile.ZipInfo("pack.json")]
    with pytest.raises(DrowseArchiveError, match="duplicate"):
        _validate_zip_infos(duplicate)
    collision = [zipfile.ZipInfo("pack.json"), zipfile.ZipInfo("PACK.JSON")]
    with pytest.raises(DrowseArchiveError, match="case-insensitively"):
        _validate_zip_infos(collision)


def test_rejects_nul_truncated_raw_zip_path() -> None:
    info = zipfile.ZipInfo("pack.json\x00ignored")
    assert info.filename == "pack.json"
    with pytest.raises(DrowseArchiveError, match="raw path"):
        _validate_zip_infos([info])


def test_rejects_symlink_special_and_encrypted_entries() -> None:
    symlink = zipfile.ZipInfo("pack.json")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(DrowseArchiveError, match="regular file"):
        _validate_zip_infos([symlink])

    special = zipfile.ZipInfo("pack.json")
    special.create_system = 3
    special.external_attr = (stat.S_IFIFO | 0o600) << 16
    with pytest.raises(DrowseArchiveError, match="regular file"):
        _validate_zip_infos([special])

    encrypted = zipfile.ZipInfo("pack.json")
    encrypted.flag_bits |= 0x1
    with pytest.raises(DrowseArchiveError, match="encrypted"):
        _validate_zip_infos([encrypted])


def test_rejects_entry_total_ratio_and_count_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.store.archive as module

    oversized = zipfile.ZipInfo("pack.json")
    oversized.file_size = 11
    oversized.compress_size = 11
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 10)
    with pytest.raises(DrowseArchiveError, match="larger"):
        _validate_zip_infos([oversized])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 100)
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_TOTAL_BYTES", 15)
    first = zipfile.ZipInfo("one")
    first.file_size = first.compress_size = 10
    second = zipfile.ZipInfo("two")
    second.file_size = second.compress_size = 10
    with pytest.raises(DrowseArchiveError, match="total"):
        _validate_zip_infos([first, second])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_TOTAL_BYTES", 1000)
    ratio = zipfile.ZipInfo("ratio")
    ratio.file_size = 101
    ratio.compress_size = 1
    with pytest.raises(DrowseArchiveError, match="100:1"):
        _validate_zip_infos([ratio])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_ENTRIES", 1)
    with pytest.raises(DrowseArchiveError, match="entries"):
        _validate_zip_infos([zipfile.ZipInfo("one"), zipfile.ZipInfo("two")])


@pytest.mark.parametrize("as_", [None, "shared/renamed"])
def test_rejects_primary_identity_mismatch_before_optional_rewrite(
    tmp_path: Path, as_: str | None,
) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        path = "manifolds/local/demo/manifold.json"
        payload = json.loads(entries[path])
        payload["name"] = "other"
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(manifold_dir("local", "demo"))
    with pytest.raises(DrowseArchiveError, match="contains manifold identity"):
        install_manifold(bad, as_=as_)


def test_rejects_template_path_identity_mismatch(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "days", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        path = "templates/local/weekday/template.json"
        payload = json.loads(entries[path])
        payload["name"] = "weekend"
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    with pytest.raises(DrowseArchiveError, match="contains identity 'weekend'"):
        install_manifold(bad)


def test_accepts_immutable_hugging_face_commit_provenance(tmp_path: Path) -> None:
    folder = _discover()
    commit = "a" * 40
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = f"hf://owner/repo@{commit}"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "demo.drowse"

    export_drowse_archive("local", "demo", output=output)

    with zipfile.ZipFile(output) as archive:
        pack = json.loads(archive.read("pack.json"))
    assert pack["source"] == {
        "uri": f"hf://owner/repo@{commit}",
        "repository": "owner/repo",
        "revision": commit,
    }


def test_ignores_entries_outside_its_closure(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    extra = tmp_path / "extra.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(_manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        entries["models/weights.bin"] = b"nope"

    _rewrite_manifest(valid, extra, mutate)
    shutil.rmtree(manifold_dir("local", "demo"))
    installed = install_manifold(extra)
    assert ManifoldFolder.load(installed).name == "demo"
    assert not (drowse_home() / "models" / "weights.bin").exists()


def test_installs_a_6_1_archive_with_retired_hash_maps(tmp_path: Path) -> None:
    """A 6.1 archive carries per-entry hashes in ``pack.json`` and a ``files``
    map in ``manifold.json`` (plus a merge-style ``tensor_sha256``); all of
    them are ignored."""
    folder = _discover()
    tensor = _fit(folder)
    current = tmp_path / "current.drowse"
    legacy = tmp_path / "legacy.drowse"
    export_drowse_archive("local", "demo", output=current)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        manifest["files"] = {
            name: {"size": len(content), "sha256": "0" * 64}
            for name, content in entries.items() if name != "pack.json"
        }
        path = "manifolds/local/demo/manifold.json"
        payload = json.loads(entries[path])
        payload["files"] = {tensor.name: "0" * 64}
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(current, legacy, mutate)
    shutil.rmtree(folder)
    installed = install_manifold(legacy)
    loaded = ManifoldFolder.load(installed)
    assert loaded.tensor_models() == [tensor.stem]
    assert "files" not in json.loads((installed / "manifold.json").read_text())


def test_rejects_missing_template_closure(tmp_path: Path) -> None:
    folder, _template = _template_manifold()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "days", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        entries.pop("templates/local/weekday/template.json")
        manifest["template"] = None

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    with pytest.raises(DrowseArchiveError, match="missing its template closure"):
        install_manifold(bad)


def test_install_leaves_no_stage_when_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.store.install as module

    folder = _discover()
    archive = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=archive)
    shutil.rmtree(folder)

    def fail_rewrite(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(module, "_rewrite_identity", fail_rewrite)
    with pytest.raises(RuntimeError, match="stage failure"):
        install_manifold(archive)

    assert not any((drowse_home() / ".staging").glob("*/manifolds"))
    assert not manifold_dir("local", "demo").exists()
    assert list(iter_manifold_folders()) == []


def test_cli_export_drowse_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from drowse.cli.main import main

    _discover()
    output = tmp_path / "cli.drowse"
    main(["pack", "export", "archive", "local/demo", "-o", str(output)])
    assert output.is_file()
    assert f"Wrote {output}" in capsys.readouterr().out


def test_install_manifold_routes_local_archive(tmp_path: Path) -> None:
    folder = _discover()
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)
    assert install_manifold(str(output)) == manifold_dir("local", "demo")


def test_install_manifold_rejects_missing_local_archive(tmp_path: Path) -> None:
    missing = tmp_path / "missing.drowse"
    with pytest.raises(FileNotFoundError, match="Drowse archive not found"):
        install_manifold(str(missing))


@pytest.mark.parametrize("coordinate", [True, float("nan"), float("inf")])
def test_authored_coordinates_must_be_finite_non_boolean(coordinate: object) -> None:
    domain = {
        "type": "box",
        "axes": [{"name": "x", "periodic": False, "lo": 0.0, "hi": 1.0}],
    }
    nodes = [
        {"label": "a", "coords": [coordinate], "statements": ["a"]},
        {"label": "b", "coords": [0.5], "statements": ["b"]},
        {"label": "c", "coords": [1.0], "statements": ["c"]},
    ]
    with pytest.raises(ManifoldFormatError, match="needs 'coords'"):
        create_manifold_folder("local", "invalid", "", domain, nodes)


@pytest.mark.parametrize("coordinate", [True, float("nan"), float("inf")])
def test_manifold_loader_rejects_nonfinite_or_boolean_coords(
    coordinate: object,
) -> None:
    domain = {
        "type": "box",
        "axes": [{"name": "x", "periodic": False, "lo": 0.0, "hi": 1.0}],
    }
    nodes = [
        {"label": "a", "coords": [0.0], "statements": ["a"]},
        {"label": "b", "coords": [0.5], "statements": ["b"]},
        {"label": "c", "coords": [1.0], "statements": ["c"]},
    ]
    folder, _ = create_manifold_folder("local", "invalid", "", domain, nodes)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["nodes"][0]["coords"] = [coordinate]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match="needs 'coords'"):
        ManifoldFolder.load(folder)
