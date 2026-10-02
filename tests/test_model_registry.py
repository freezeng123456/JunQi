"""Registry checks read real files and never deserialize model checkpoints."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_model_registry import audit_registry, main


@pytest.fixture
def registry_files(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact = root / "model.pt"
    artifact.write_bytes(b"checkpoint contents that do not need torch to audit\x00\xff")
    registry = {
        "schema_version": 1,
        "default_model_id": "facts-final",
        "artifact_roots": {"training": "Recovered training files"},
        "models": [{
            "id": "facts-final",
            "notes": "Additional provenance fields are permitted",
            "artifacts": [{
                "role": "checkpoint", "root": "training", "path": "model.pt",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }],
        }],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path, registry, root, artifact


def _write_registry(path, registry):
    path.write_text(json.dumps(registry), encoding="utf-8")


def test_registry_verifies_actual_file_and_preserves_input(registry_files):
    path, registry, root, artifact = registry_files
    original_registry, original_artifact = path.read_bytes(), artifact.read_bytes()
    result = audit_registry(path, {"training": root})
    assert result["ok"] and result["errors"] == []
    assert result["models_checked"] == result["artifacts_checked"] == 1
    assert result["artifacts"][0]["actual_sha256"] == registry["models"][0]["artifacts"][0]["sha256"]
    assert path.read_bytes() == original_registry
    assert artifact.read_bytes() == original_artifact


def test_registry_detects_modified_artifact(registry_files):
    path, _, root, artifact = registry_files
    artifact.write_bytes(b"changed weights")
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert "SHA-256 mismatch" in "\n".join(result["errors"])


def test_missing_artifact_is_failure(registry_files):
    path, _, root, artifact = registry_files
    artifact.unlink()
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert "artifact file is missing" in "\n".join(result["errors"])


def test_missing_root_cannot_pass_as_unverified(registry_files):
    path, _, _, _ = registry_files
    result = audit_registry(path, {})
    assert not result["ok"]
    assert "provide --artifact-root training=PATH" in "\n".join(result["errors"])


@pytest.mark.parametrize("relative", ["../outside.pt", "/outside.pt", "C:/outside.pt", "..\\outside.pt"])
def test_path_traversal_is_rejected(registry_files, relative):
    path, registry, root, _ = registry_files
    registry["models"][0]["artifacts"][0]["path"] = relative
    _write_registry(path, registry)
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert "relative file path without traversal" in "\n".join(result["errors"])


def test_symlink_cannot_escape_root(registry_files, tmp_path):
    path, registry, root, artifact = registry_files
    outside = tmp_path / "outside.pt"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert "escapes artifact root through a symlink" in "\n".join(result["errors"])


def test_duplicate_model_id_is_rejected(registry_files):
    path, registry, root, _ = registry_files
    registry["models"].append(deepcopy(registry["models"][0]))
    _write_registry(path, registry)
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert "Duplicate model id 'facts-final'" in "\n".join(result["errors"])


@pytest.mark.parametrize(("field", "bad_value", "message"), [
    ("schema_version", True, "schema_version"),
    ("default_model_id", "missing", "does not identify a model"),
    ("models", [], "models must be a nonempty list"),
    ("artifact_roots", {"training": 4}, "name and description"),
])
def test_top_level_schema_is_checked(registry_files, field, bad_value, message):
    path, registry, root, _ = registry_files
    registry[field] = bad_value
    _write_registry(path, registry)
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert message in "\n".join(result["errors"])


@pytest.mark.parametrize(("field", "bad_value", "message"), [
    ("role", "", "role must be a nonempty string"),
    ("root", "missing", "is not declared"),
    ("path", [], "path must be a nonempty string"),
    ("sha256", "abc", "64 hexadecimal characters"),
])
def test_artifact_schema_is_checked(registry_files, field, bad_value, message):
    path, registry, root, _ = registry_files
    registry["models"][0]["artifacts"][0][field] = bad_value
    _write_registry(path, registry)
    result = audit_registry(path, {"training": root})
    assert not result["ok"]
    assert message in "\n".join(result["errors"])


def test_cli_stdlib_only_supports_multiple_roots_and_json_report(registry_files, tmp_path):
    path, registry, root, artifact = registry_files
    registry["artifact_roots"]["audit"] = "Local audit artifacts"
    registry["models"][0]["artifacts"].append({
        **registry["models"][0]["artifacts"][0], "root": "audit", "role": "copy",
    })
    _write_registry(path, registry)
    output = tmp_path / "reports" / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_model_registry.py"
    # -S disables site packages, proving that neither torch nor any other
    # installed dependency is needed to audit the byte-level model registry.
    run = subprocess.run([
        sys.executable, "-S", str(script), "--registry", str(path),
        "--artifact-root", f"training={root}", "--artifact-root", f"audit={root}",
        "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == json.loads(run.stdout)
    assert report["ok"] and report["artifacts_checked"] == 2
    assert artifact.is_file()


def test_cli_missing_root_exits_nonzero(registry_files, capsys):
    path, _, _, _ = registry_files
    assert main(["--registry", str(path)]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


@pytest.mark.parametrize("target", ["registry", "artifact"])
def test_report_cannot_overwrite_inputs(registry_files, target):
    path, _, root, artifact = registry_files
    output = path if target == "registry" else artifact
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--registry", str(path), "--artifact-root", f"training={root}", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_bad_json_reports_read_failure(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not json", encoding="utf-8")
    result = audit_registry(path, {})
    assert not result["ok"]
    assert "Cannot read registry" in "\n".join(result["errors"])
