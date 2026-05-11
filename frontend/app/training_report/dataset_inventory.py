from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".tif", ".tiff"}
MLMARKUP_CANDIDATES = (
    "/data/MLMarkup",
    "/data/mlsystem/MLMarkup",
    "/data/mlsystem/markup",
    "/opt/mlsystem/MLMarkup",
)


@dataclass(frozen=True)
class ClassSpec:
    class_name: str
    class_slug: str
    aliases: tuple[str, ...] = ()


SUPPORTED_CLASSES: tuple[ClassSpec, ...] = (
    ClassSpec("Озера", "lakes", ("lake", "lakes")),
    ClassSpec("Абразия", "abrasion", ("abrasion",)),
    ClassSpec("Обвально-оползневые и осыпные", "obval_opolz_osyp", ("landslide", "оползневые", "осыпные")),
    ClassSpec("Водная эрозия", "water_erosion", ("water_erosion",)),
    ClassSpec("Ветровая эрозия", "wind_erosion", ("wind_erosion",)),
    ClassSpec("Вырубки", "deforest", ("deforestation", "cuttings", "deforest")),
    ClassSpec("Гари", "burnt_forests", ("burnt_forests", "burned", "fire")),
    ClassSpec("Границы леса", "forest_boundaries", ("forest", "forest_boundaries")),
    ClassSpec("Карьеры", "quarries", ("careers", "quarries")),
    ClassSpec("Опустынивание", "desertification", ("desertification",)),
    ClassSpec("Пашни", "arable_land", ("areas_of_used_arable_land", "arable_land", "pashni")),
)


def class_specs_by_slug() -> dict[str, ClassSpec]:
    return {item.class_slug: item for item in SUPPORTED_CLASSES}


def class_specs_by_name() -> dict[str, ClassSpec]:
    return {item.class_name: item for item in SUPPORTED_CLASSES}


def class_slug_for_text(text: str) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for spec in SUPPORTED_CLASSES:
        candidates = (spec.class_name, spec.class_slug, *spec.aliases)
        if any(_candidate_matches(normalized, _normalize_text(candidate)) for candidate in candidates):
            return spec.class_slug
    return None


def discover_mlmarkup_path(configured: Path | str | None = None) -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    env_value = os.getenv("MLSYSTEM_MLMARKUP_REPO_PATH") or os.getenv("MLMARKUP_REPO_PATH")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(Path(item) for item in MLMARKUP_CANDIDATES)
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def inventory_all_classes(mlmarkup_path: Path | None, *, images_uri: str = "") -> dict[str, dict[str, Any]]:
    return {
        spec.class_slug: inventory_class(mlmarkup_path, spec, images_uri=images_uri)
        for spec in SUPPORTED_CLASSES
    }


def inventory_class(mlmarkup_path: Path | None, spec: ClassSpec, *, images_uri: str = "") -> dict[str, Any]:
    base = {
        "class_name": spec.class_name,
        "class_slug": spec.class_slug,
        "class_dir": None,
        "geojson_files": [],
        "txt_files": [],
        "dataset_fingerprint": None,
        "dataset_date": None,
        "dataset_date_source": "unknown",
        "objects_count": 0,
        "scenes_count": 0,
        "scenes_count_source": "unknown",
        "warnings": [],
    }
    if not mlmarkup_path:
        base["warnings"].append("MLMarkup path is not available")
        return base
    class_dir = find_class_dir(mlmarkup_path, spec)
    if not class_dir:
        base["warnings"].append("class directory is not available in MLMarkup")
        return base

    geojson_files = sorted(class_dir.glob("*.geojson"))
    txt_files = sorted(class_dir.glob("*.txt"))
    warnings: list[str] = []
    objects_count = sum(count_geojson_objects(path, warnings=warnings) for path in geojson_files)
    scenes_count, scenes_source, scene_warnings = count_scene_files(
        txt_files,
        class_dir=class_dir,
        mlmarkup_path=mlmarkup_path,
        images_uri=images_uri,
    )
    warnings.extend(scene_warnings)
    dataset_date, date_source = dataset_publication_date(mlmarkup_path, class_dir, [*geojson_files, *txt_files])
    fingerprint = dataset_fingerprint(mlmarkup_path, class_dir, [*geojson_files, *txt_files])

    base.update(
        {
            "class_dir": str(class_dir),
            "geojson_files": [str(path) for path in geojson_files],
            "txt_files": [str(path) for path in txt_files],
            "dataset_fingerprint": fingerprint,
            "dataset_date": dataset_date,
            "dataset_date_source": date_source,
            "objects_count": objects_count,
            "scenes_count": scenes_count,
            "scenes_count_source": scenes_source,
            "warnings": warnings,
        }
    )
    return base


def find_class_dir(mlmarkup_path: Path, spec: ClassSpec) -> Path | None:
    exact = mlmarkup_path / spec.class_name
    if exact.exists() and exact.is_dir():
        return exact
    normalized_names = {_normalize_text(spec.class_name), _normalize_text(spec.class_slug), *(_normalize_text(alias) for alias in spec.aliases)}
    try:
        for child in mlmarkup_path.iterdir():
            if child.is_dir() and _normalize_text(child.name) in normalized_names:
                return child
    except OSError:
        return None
    return None


def count_geojson_objects(path: Path, *, warnings: list[str] | None = None) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(f"{path.name}: cannot parse GeoJSON ({type(exc).__name__})")
        return 0
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        return len(features) if isinstance(features, list) else 0
    if isinstance(payload, dict) and payload.get("type") == "Feature":
        return 1
    if warnings is not None:
        warnings.append(f"{path.name}: unsupported GeoJSON type")
    return 0


def count_scene_files(
    txt_files: list[Path],
    *,
    class_dir: Path,
    mlmarkup_path: Path,
    images_uri: str = "",
) -> tuple[int, str, list[str]]:
    warnings: list[str] = []
    total = 0
    used_dir_expansion = False
    for txt in txt_files:
        try:
            entries = _read_scene_entries(txt)
        except OSError as exc:
            warnings.append(f"{txt.name}: cannot read scenes file ({type(exc).__name__})")
            continue
        for entry in entries:
            count, source, warning = _count_scene_entry(entry, class_dir=class_dir, mlmarkup_path=mlmarkup_path, images_uri=images_uri)
            total += count
            used_dir_expansion = used_dir_expansion or source == "txt_dirs_expanded"
            if warning:
                warnings.append(warning)
    if not txt_files:
        return 0, "unknown", ["no scenes txt files found"]
    return total, "txt_dirs_expanded" if used_dir_expansion else "txt_files", warnings


def dataset_publication_date(mlmarkup_path: Path, class_dir: Path, files: list[Path]) -> tuple[str | None, str]:
    git_date = _git_latest_commit_date(mlmarkup_path, class_dir)
    if git_date:
        return git_date[:10], "git_commit"
    mtimes = []
    for path in [class_dir, *files]:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    if not mtimes:
        return None, "unknown"
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).date().isoformat(), "file_mtime"


def dataset_fingerprint(mlmarkup_path: Path, class_dir: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(str(class_dir.relative_to(mlmarkup_path) if _is_relative_to(class_dir, mlmarkup_path) else class_dir).encode("utf-8"))
    commit = git_commit(mlmarkup_path)
    if commit:
        digest.update(commit.encode("utf-8"))
    for path in sorted(files):
        digest.update(str(path.name).encode("utf-8"))
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def git_commit(repo_path: Path) -> str | None:
    result = _run_git(repo_path, ["rev-parse", "HEAD"])
    return result if result else None


def _read_scene_entries(path: Path) -> list[str]:
    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _count_scene_entry(entry: str, *, class_dir: Path, mlmarkup_path: Path, images_uri: str = "") -> tuple[int, str, str | None]:
    clean_entry = entry.strip().strip("\"'")
    if _looks_like_image_file(clean_entry):
        return 1, "txt_files", None
    for base in (Path(clean_entry), class_dir / clean_entry, mlmarkup_path / clean_entry):
        if base.exists():
            if base.is_file():
                return 1, "txt_files", None
            if base.is_dir():
                return _count_local_images(base), "txt_dirs_expanded", None
    if clean_entry.startswith("s3://") or images_uri.startswith("s3://"):
        count = _count_s3_images(clean_entry, images_uri=images_uri)
        if count is not None:
            return count, "txt_dirs_expanded", None
    return 1, "txt_files", f"{entry}: ambiguous scene entry; counted as one scene"


def _count_local_images(directory: Path) -> int:
    count = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def _count_s3_images(entry: str, *, images_uri: str = "") -> int | None:
    try:
        import boto3
        from urllib.parse import urlparse
    except Exception:  # noqa: BLE001
        return None
    uri = entry if entry.startswith("s3://") else images_uri.rstrip("/") + "/" + entry.lstrip("/")
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        return None
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/") and Path(prefix).suffix.lower() in IMAGE_EXTENSIONS:
        return 1
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("MLSYSTEM_S3_ENDPOINT_URL")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )
    total = 0
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": parsed.netloc, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        payload = client.list_objects_v2(**kwargs)
        total += sum(1 for item in payload.get("Contents") or [] if Path(str(item.get("Key", ""))).suffix.lower() in IMAGE_EXTENSIONS)
        if not payload.get("IsTruncated"):
            return total
        token = payload.get("NextContinuationToken")
        if not token:
            return total


def _looks_like_image_file(value: str) -> bool:
    suffix = Path(value).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def _git_latest_commit_date(repo_path: Path, target: Path) -> str | None:
    try:
        target_arg = str(target.relative_to(repo_path))
    except ValueError:
        target_arg = str(target)
    return _run_git(repo_path, ["log", "-1", "--format=%cI", "--", target_arg])


def _run_git(repo_path: Path, args: list[str]) -> str | None:
    if not (repo_path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_path}", "-C", str(repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_text(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _candidate_matches(text: str, candidate: str) -> bool:
    if not candidate:
        return False
    if candidate == text:
        return True
    if len(candidate) < 4 and len(text) < 4:
        return False
    return candidate in text or text in candidate
