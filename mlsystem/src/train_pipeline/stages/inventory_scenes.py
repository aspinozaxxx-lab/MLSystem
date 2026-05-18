from __future__ import annotations

from typing import Any

from ...dataset_preparing.api import build_scene_matching_report, parse_scene_list_text
from ...settings.api import load_config
from ...storage.api import build_s3_layout_status, find_layout_files, list_s3_objects, read_s3_text, write_json
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    checks: list[StageCheck] = []
    warnings: list[str] = []
    errors: list[str] = []
    artifacts: dict[str, str] = {}

    pipeline_config = load_config()
    checks.append(StageCheck("config", "ok", f"experiment_id={ctx.config.experiment_id}"))

    try:
        s3_layout = build_s3_layout_status(pipeline_config)
        checks.append(StageCheck("layout", "ok", "S3/MinIO layout status collected"))
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", checks, errors=[f"S3 layout check failed: {exc}"])
        raise StageFailure("S3 layout check failed", report) from exc

    try:
        images = list_s3_objects(pipeline_config, ctx.config.images_uri, suffixes=(".tif", ".tiff"))
        checks.append(StageCheck("images_uri", "ok", f"{len(images)} TIFF/TIF objects listed"))
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", checks, errors=[f"images_uri is not readable: {exc}"])
        raise StageFailure("images_uri is not readable", report) from exc

    try:
        annotation_uri, scenes_uri = find_layout_files(
            pipeline_config,
            ctx.config.layout_uri,
            ctx.config.scenes_file,
            ctx.config.annotation_file,
        )
        checks.append(StageCheck("layout_files", "ok", f"scenes_file={ctx.config.scenes_file}, annotation_file found"))
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", checks, errors=[f"Required layout file is missing: {exc}"])
        raise StageFailure("required layout file is missing", report) from exc

    try:
        entries = parse_scene_list_text(read_s3_text(pipeline_config, scenes_uri))
        checks.append(StageCheck("scenes_file", "ok", f"{len(entries)} scene rows read"))
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", checks, errors=[f"scenes_file cannot be read: {exc}"])
        raise StageFailure("scenes_file cannot be read", report) from exc

    preferred_prefixes = list(ctx.config.preprocess.get("scene_matching_prefer_prefixes") or [])
    matching = {
        **build_scene_matching_report(entries, images, preferred_key_prefixes=preferred_prefixes),
        "images_uri": ctx.config.images_uri,
        "layout_uri": ctx.config.layout_uri,
        "annotation_uri": annotation_uri,
        "scenes_uri": scenes_uri,
    }
    matched = matching.get("matched") or []
    missing = matching.get("missing") or []
    ambiguous = matching.get("ambiguous") or []
    if int(matching.get("total_tif_images_available") or 0) > len(matched):
        warnings.append(
            f"{int(matching.get('total_tif_images_available') or 0) - len(matched)} TIFF/TIF images are available but not selected by scenes_file"
        )
    if missing:
        errors.append(f"{len(missing)} scenes from scenes_file were not matched to TIFF/TIF images: {missing[:10]}")
    if ambiguous:
        errors.append(f"{len(ambiguous)} scenes have ambiguous image matches")
    if not matched:
        errors.append("No scenes were matched")

    inventory = {
        "schema_version": 1,
        "stage": ctx.stage_id,
        "experiment_id": ctx.config.experiment_id,
        "annotation_source": (getattr(ctx.config, "annotations", None) or {}).get("source") or "layout_uri",
        "annotations": getattr(ctx.config, "annotations", None) or {},
        "images_uri": ctx.config.images_uri,
        "layout_uri": ctx.config.layout_uri,
        "annotation_uri": annotation_uri,
        "scenes_uri": scenes_uri,
        "scene_count": int(matching.get("expanded_scene_count") or len(matched)),
        "requested_entries_count": int(matching.get("requested_entries_count") or len(entries)),
        "requested_files_count": int(matching.get("requested_files_count") or 0),
        "requested_folders_count": int(matching.get("requested_folders_count") or 0),
        "expanded_scene_count": int(matching.get("expanded_scene_count") or len(matched)),
        "folder_expansions": matching.get("folder_expansions") or {},
        "unresolved_entries": matching.get("unresolved_entries") or missing,
        "ambiguous_entries": ambiguous,
        "scene_file_entries_sample": entries[:20],
        "available_tiff_samples": matching.get("available_tiff_samples") or [],
        "matched_count": len(matched),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "available_image_count": len(images),
        "available_images": _compact_images(images),
        "matched": matched,
        "missing": missing,
        "ambiguous": ambiguous,
    }

    inventory_path = ctx.store.run_dir / "inventory_scenes.json"
    report_path = ctx.store.run_dir / "scene_inventory_report.txt"
    matched_path = ctx.store.run_dir / "matched_scenes.txt"
    missing_path = ctx.store.run_dir / "missing_scenes.txt"
    compatibility_path = ctx.store.run_dir / "scene_matching_report.json"
    write_json(inventory_path, inventory)
    write_json(compatibility_path, matching)
    matched_path.write_text("\n".join(item.get("entry", "") for item in matched) + ("\n" if matched else ""), encoding="utf-8")
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    report_path.write_text(_inventory_report_text(inventory, warnings, errors), encoding="utf-8")
    artifacts.update(
        {
            "inventory_scenes.json": str(inventory_path),
            "scene_inventory_report.txt": str(report_path),
            "matched_scenes.txt": str(matched_path),
            "missing_scenes.txt": str(missing_path),
            "scene_matching_report.json": str(compatibility_path),
        }
    )
    ctx.store.update_summary(scene_inventory=inventory, scene_matching=matching)

    checks.append(
        StageCheck(
            "files found",
            "failed" if errors else "ok",
            f"matched={len(matched)}, missing={len(missing)}, ambiguous={len(ambiguous)}",
        )
    )
    counters = {
        "scene_rows": int(matching.get("expanded_scene_count") or len(matched)),
        "requested_entries_count": int(matching.get("requested_entries_count") or len(entries)),
        "requested_files_count": int(matching.get("requested_files_count") or 0),
        "requested_folders_count": int(matching.get("requested_folders_count") or 0),
        "expanded_scene_count": int(matching.get("expanded_scene_count") or len(matched)),
        "matched_scenes": len(matched),
        "missing_scenes": len(missing),
        "ambiguous_scenes": len(ambiguous),
        "available_images": len(images),
    }
    report = StageReport(ctx.stage_id, "failed" if errors else "success", checks, counters, warnings, errors, artifacts)
    if errors:
        raise StageFailure("inventory_scenes failed", report)
    return report


def _compact_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"bucket": item.get("bucket"), "key": item.get("key"), "name": item.get("name"), "size": item.get("size")} for item in images]


def _inventory_report_text(inventory: dict[str, Any], warnings: list[str], errors: list[str]) -> str:
    lines = [
        "inventory_scenes",
        f"images_uri={inventory.get('images_uri')}",
        f"layout_uri={inventory.get('layout_uri')}",
        f"scenes_uri={inventory.get('scenes_uri')}",
        f"annotation_uri={inventory.get('annotation_uri')}",
        f"scene_count={inventory.get('scene_count')}",
        f"requested_entries_count={inventory.get('requested_entries_count')}",
        f"requested_files_count={inventory.get('requested_files_count')}",
        f"requested_folders_count={inventory.get('requested_folders_count')}",
        f"expanded_scene_count={inventory.get('expanded_scene_count')}",
        f"matched_count={inventory.get('matched_count')}",
        f"missing_count={inventory.get('missing_count')}",
        f"ambiguous_count={inventory.get('ambiguous_count')}",
        "",
        "[scene_file_entries_sample]",
        *[str(item) for item in inventory.get("scene_file_entries_sample") or []],
        "",
        "[available_tiff_samples]",
        *[str(item) for item in inventory.get("available_tiff_samples") or []],
        "",
        "[folder_expansions]",
        *[
            f"{raw} -> {payload.get('matched_folder')} ({payload.get('scene_count')} scenes)"
            for raw, payload in sorted((inventory.get("folder_expansions") or {}).items())
        ],
        "",
        "[missing]",
        *(inventory.get("missing") or []),
        "",
        "[warnings]",
        *warnings,
        "",
        "[errors]",
        *errors,
    ]
    return "\n".join(lines) + "\n"
