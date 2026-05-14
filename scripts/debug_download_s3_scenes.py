from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.pipeline_config import load_config
from mlsystem.src.storage import s3 as s3_storage


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a few S3 GeoTIFF scenes for local virtual sampling debug.")
    parser.add_argument("--manifest", required=True, help="Local or s3:// dataset_manifest.json.")
    parser.add_argument("--images-uri", required=True, help="S3 image prefix, for example s3://mlsystems/images/.")
    parser.add_argument("--output-dir", default=r"E:\Projects\NSPD\Images\test", help="Local output directory.")
    parser.add_argument("--max-scenes", type=int, default=3, help="Maximum scenes to download.")
    parser.add_argument("--prefer-positive", default="true", help="Prefer scenes with positive object counts: true/false.")
    args = parser.parse_args()

    config = load_config()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(config, args.manifest)
    rows = _select_manifest_rows(manifest, max_scenes=max(1, int(args.max_scenes)), prefer_positive=_as_bool(args.prefer_positive))
    if not rows:
        raise RuntimeError("No train_scenes or val_scenes were found in the manifest")

    try:
        client = s3_storage.s3_client(config)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "S3 credentials are not available. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "or configure the mc alias used by MLSYSTEM_PIPELINE_CONFIG. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    downloaded: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for row in rows:
        bucket, key = _resolve_s3_object(args.images_uri, row)
        target = output_dir / PurePosixPath(key).name
        selected.append(
            {
                "scene_name": row.get("name") or row.get("entry") or PurePosixPath(key).name,
                "split": row.get("_split"),
                "object_count": row.get("_object_count"),
                "s3_uri": f"s3://{bucket}/{key}",
                "local_path": str(target),
            }
        )
    print(json.dumps({"selected_scenes": selected}, ensure_ascii=False, indent=2))
    for item in selected:
        bucket, key = s3_storage.s3_parts(str(item["s3_uri"]))
        target = Path(str(item["local_path"]))
        print(f"Downloading s3://{bucket}/{key} -> {target}")
        try:
            client.download_file(bucket, key, str(target))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download s3://{bucket}/{key}. Check credentials, endpoint, bucket, and key. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        downloaded.append({**item, "bucket": bucket, "key": key, "path": str(target), "size_bytes": target.stat().st_size})

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "selected_scenes": selected, "downloaded": downloaded}, ensure_ascii=False, indent=2))
    return 0


def _resolve_output_dir(raw: str) -> Path:
    if platform.system().lower() != "windows":
        wsl_path = Path("/mnt/e/Projects/NSPD/Images/test")
        if wsl_path.exists():
            return wsl_path
    return Path(raw)


def _read_manifest(config: Any, manifest: str) -> dict[str, Any]:
    if manifest.startswith("s3://"):
        return s3_storage.read_s3_json(config, manifest)
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def _select_manifest_rows(manifest: dict[str, Any], *, max_scenes: int, prefer_positive: bool) -> list[dict[str, Any]]:
    counts = {
        str(item.get("scene_name") or item.get("name") or item.get("entry") or item.get("key") or ""): int(item.get("object_count") or 0)
        for item in manifest.get("scene_object_counts") or []
    }

    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val"):
        for item in manifest.get(f"{split_name}_scenes") or []:
            row = dict(item)
            row["_split"] = split_name
            row["_object_count"] = _object_count_for_row(row, counts)
            rows.append(row)
    if not prefer_positive:
        return rows[:max_scenes]

    def score(row: dict[str, Any]) -> tuple[int, str]:
        count = int(row.get("_object_count") or 0)
        return (1 if count > 0 else 0, str(row.get("name") or row.get("entry") or row.get("key") or ""))

    rows.sort(key=score, reverse=True)
    return rows[:max_scenes]


def _object_count_for_row(row: dict[str, Any], counts: dict[str, int]) -> int | None:
    identities = [
        str(row.get("entry") or ""),
        str(row.get("name") or ""),
        str(row.get("key") or ""),
        PurePosixPath(str(row.get("key") or row.get("name") or row.get("entry") or "")).name,
    ]
    values = [counts[identity] for identity in identities if identity in counts]
    return max(values) if values else None


def _resolve_s3_object(images_uri: str, row: dict[str, Any]) -> tuple[str, str]:
    if str(row.get("key") or "").startswith("s3://"):
        return s3_storage.s3_parts(str(row["key"]))
    bucket, prefix = s3_storage.s3_parts(images_uri)
    key = str(row.get("key") or "").strip().replace("\\", "/")
    if key and "/" in key:
        return bucket, key
    name = key or str(row.get("name") or row.get("entry") or "").strip().replace("\\", "/")
    if not name:
        raise RuntimeError(f"Cannot resolve S3 key for manifest row: {row}")
    return bucket, f"{prefix.rstrip('/')}/{PurePosixPath(name).name}" if prefix else PurePosixPath(name).name


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
