"""Validate a model registry and verify its artifacts without loading weights.

Only the Python standard library is needed. Artifact roots are explicit local
mounts: descriptions in the registry never grant access to a guessed location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_registry(
    registry_path: str | Path,
    artifact_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Return a JSON report; any malformed or unverified artifact makes it fail."""
    registry_path = Path(registry_path).expanduser().resolve()
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "registry": str(registry_path),
        "ok": False,
        "models_checked": 0,
        "artifacts_checked": 0,
        "artifacts": [],
        "errors": errors,
    }
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read registry: {exc}")
        return report
    if not isinstance(registry, dict):
        errors.append("Registry must be a JSON object")
        return report
    if type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
        errors.append("schema_version must be the integer 1")
    default_model_id = registry.get("default_model_id")
    if not _nonempty_string(default_model_id):
        errors.append("default_model_id must be a nonempty string")
    declared_roots = registry.get("artifact_roots")
    if not isinstance(declared_roots, dict) or not declared_roots:
        errors.append("artifact_roots must be a nonempty object of names and descriptions")
        declared_roots = {}
    for name, description in declared_roots.items():
        if not _nonempty_string(name) or not _nonempty_string(description):
            errors.append("Each artifact_roots name and description must be a nonempty string")

    resolved_roots: dict[str, Path] = {}
    for name, raw_path in artifact_roots.items():
        if name not in declared_roots:
            errors.append(f"Supplied artifact root {name!r} is not declared in the registry")
            continue
        try:
            root = Path(raw_path).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("not a directory")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"Artifact root {name!r} is unavailable: {exc}")
        else:
            resolved_roots[name] = root

    models = registry.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a nonempty list")
        return report
    seen_ids: set[str] = set()
    for model_index, model in enumerate(models):
        report["models_checked"] += 1
        if not isinstance(model, dict):
            errors.append(f"models[{model_index}] must be an object")
            continue
        model_id = model.get("id")
        model_label = f"models[{model_index}]"
        if not _nonempty_string(model_id):
            errors.append(f"{model_label}.id must be a nonempty string")
        elif model_id in seen_ids:
            errors.append(f"Duplicate model id {model_id!r}")
        else:
            seen_ids.add(model_id)
            model_label = f"model {model_id!r}"
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{model_label}.artifacts must be a nonempty list")
            continue
        for artifact_index, artifact in enumerate(artifacts):
            report["artifacts_checked"] += 1
            label = f"{model_label}.artifacts[{artifact_index}]"
            result: dict[str, Any] = {
                "model_id": model_id, "artifact_index": artifact_index,
                "ok": False, "errors": [],
            }
            report["artifacts"].append(result)
            artifact_errors = result["errors"]
            if not isinstance(artifact, dict):
                artifact_errors.append("must be an object")
            else:
                for key in ("role", "root", "path", "sha256"):
                    result[key] = artifact.get(key)
                    if not _nonempty_string(artifact.get(key)):
                        artifact_errors.append(f"{key} must be a nonempty string")
                if not artifact_errors:
                    name, relative = artifact["root"], artifact["path"]
                    if name not in declared_roots:
                        artifact_errors.append(f"root {name!r} is not declared in artifact_roots")
                    elif name not in resolved_roots:
                        artifact_errors.append(
                            f"root {name!r} was not supplied or is unavailable; "
                            f"provide --artifact-root {name}=PATH"
                        )
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", artifact["sha256"]):
                        artifact_errors.append("sha256 must be exactly 64 hexadecimal characters")
                    # Registry paths use one portable syntax. Reject traversal
                    # even if it happens to resolve back inside the root.
                    pure_path = PurePosixPath(relative)
                    if (pure_path.is_absolute() or PureWindowsPath(relative).drive
                            or "\\" in relative or ".." in pure_path.parts
                            or relative == "." or "\x00" in relative):
                        artifact_errors.append("path must be a relative file path without traversal")
                    if not artifact_errors:
                        try:
                            root = resolved_roots[name]
                            path = (root / relative).resolve()
                            if not path.is_relative_to(root):
                                raise ValueError("path escapes artifact root through a symlink")
                            result["resolved_path"] = str(path)
                            if not path.is_file():
                                raise ValueError("artifact file is missing or is not a regular file")
                            result["actual_sha256"] = _sha256(path)
                            if result["actual_sha256"] != artifact["sha256"].lower():
                                artifact_errors.append("SHA-256 mismatch")
                        except (OSError, RuntimeError, ValueError) as exc:
                            artifact_errors.append(str(exc))
            result["ok"] = not artifact_errors
            errors.extend(f"{label}: {message}" for message in artifact_errors)
    if _nonempty_string(default_model_id) and default_model_id not in seen_ids:
        errors.append(f"default_model_id {default_model_id!r} does not identify a model")
    report["ok"] = not errors
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="docs/models/registry.json")
    parser.add_argument("--artifact-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, help="Optional JSON audit report")
    args = parser.parse_args(argv)
    roots: dict[str, str] = {}
    for specification in args.artifact_root:
        name, separator, path = specification.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("--artifact-root must be NAME=PATH")
        if name in roots:
            parser.error(f"duplicate --artifact-root {name!r}")
        roots[name] = path
    report = audit_registry(args.registry, roots)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        protected = [Path(report["registry"])] + [
            Path(artifact["resolved_path"]) for artifact in report["artifacts"]
            if "resolved_path" in artifact
        ]
        for path in protected:
            if output == path or (output.exists() and path.exists() and output.samefile(path)):
                parser.error("--output must not overwrite the registry or an audited artifact")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot write audit report: {exc}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
