from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

RASTER_SUFFIXES = (".tif", ".tiff")


@dataclass
class SceneMatch:
    entry: str
    key: str
    name: str
    score: float


def parse_scene_list_text(text: str | bytes) -> list[str]:
    payload = text.decode("utf-8-sig", errors="replace") if isinstance(text, bytes) else text
    entries: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line.split()[0].replace("\\", "/"))
    return entries


def norm_scene_name(value: str) -> str:
    name = PurePosixPath(value.strip()).name.lower()
    name = re.sub(r"\.aux\.xml$", "", name)
    name = re.sub(r"\.(tif|tiff)$", "", name)
    name = re.sub(r"[_\-. ]?cog$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def scene_signature(normalized: str) -> str | None:
    match = re.search(r"kanopus(\d{8})(\d{6}).*?scn(\d{1,2})", normalized)
    if not match:
        return None
    date, tm, scn = match.groups()
    return f"kanopus:{date}:{tm}:scn{int(scn):02d}"


def scene_score(needle: str, candidate: str) -> tuple[float, str]:
    if not needle or not candidate:
        return 0.0, "empty"
    if needle == candidate:
        return 1.0, "normalized_exact"
    needle_sig = scene_signature(needle)
    candidate_sig = scene_signature(candidate)
    if needle_sig and candidate_sig and needle_sig == candidate_sig:
        return 0.995, "kanopus_datetime_scn_signature"
    if len(needle) >= 16 and (needle in candidate or candidate in needle):
        return 0.98, "normalized_substring"
    return difflib.SequenceMatcher(None, needle, candidate).ratio(), "sequence_ratio"


def build_scene_matching_report(
    entries: list[str],
    images: list[dict[str, Any]],
    *,
    accept_threshold: float = 0.92,
    ambiguous_margin: float = 0.015,
    preferred_key_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    normalized_images = [(item, norm_scene_name(_image_name(item))) for item in images]
    folder_index = _build_folder_index(images)
    matched: list[SceneMatch] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    matched_identities: set[str] = set()
    requested_files: list[str] = []
    requested_folders: list[str] = []
    folder_expansions: dict[str, dict[str, Any]] = {}

    for entry in entries:
        needle = norm_scene_name(entry)
        scored: list[dict[str, Any]] = []
        for image, image_norm in normalized_images:
            score, score_reason = scene_score(needle, image_norm)
            scored.append(
                {
                    "image": image,
                    "normalized_image": image_norm,
                    "score": round(float(score), 6),
                    "reason": score_reason,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None
        exact_candidates = [item for item in scored if item["score"] == 1.0 and item["reason"] == "normalized_exact"]
        decision = "missing"
        reason = "no_reliable_candidate"
        row_extra: dict[str, Any] = {}
        if best and best["score"] >= accept_threshold:
            if len(exact_candidates) == 1:
                decision = "matched"
                reason = exact_candidates[0]["reason"]
                image = exact_candidates[0]["image"]
                _append_match_once(matched, matched_identities, SceneMatch(entry=entry, key=_image_key(image), name=_image_name(image), score=1.0))
                requested_files.append(entry)
            elif len(exact_candidates) > 1 and preferred_key_prefixes:
                preferred = []
                for prefix in preferred_key_prefixes:
                    preferred = [
                        item
                        for item in exact_candidates
                    if _image_key(item["image"]).startswith(prefix)
                    ]
                    if preferred:
                        break
                if len(preferred) == 1:
                    decision = "matched"
                    reason = "preferred_exact_duplicate"
                    image = preferred[0]["image"]
                    _append_match_once(matched, matched_identities, SceneMatch(entry=entry, key=_image_key(image), name=_image_name(image), score=1.0))
                    requested_files.append(entry)
                else:
                    decision = "ambiguous"
                    reason = "multiple_close_candidates"
                    ambiguous.append(
                        {
                            "entry": entry,
                            "normalized_scene_line": needle,
                            "candidates": [
                                {
                                    "name": _image_name(item["image"]),
                                    "key": _image_key(item["image"]),
                                    "score": item["score"],
                                    "reason": item["reason"],
                                }
                                for item in scored[:10]
                                if item["score"] >= accept_threshold
                            ],
                        }
                    )
            elif second and second["score"] >= accept_threshold and (best["score"] - second["score"]) <= ambiguous_margin:
                decision = "ambiguous"
                reason = "multiple_close_candidates"
                ambiguous.append(
                    {
                        "entry": entry,
                        "normalized_scene_line": needle,
                        "candidates": [
                            {
                            "name": _image_name(item["image"]),
                            "key": _image_key(item["image"]),
                                "score": item["score"],
                                "reason": item["reason"],
                            }
                            for item in scored[:10]
                            if item["score"] >= accept_threshold
                        ],
                    }
                )
            else:
                decision = "matched"
                reason = best["reason"]
                image = best["image"]
                _append_match_once(matched, matched_identities, SceneMatch(entry=entry, key=_image_key(image), name=_image_name(image), score=round(float(best["score"]), 4)))
                requested_files.append(entry)
        else:
            folder_result = _match_folder_entry(entry, folder_index)
            if folder_result["status"] == "matched":
                requested_folders.append(entry)
                folder_images = folder_result["images"]
                added = 0
                for image in folder_images:
                    added += int(
                        _append_match_once(
                            matched,
                            matched_identities,
                            SceneMatch(
                                entry=_canonical_scene_entry(image),
                                key=_image_key(image),
                                name=_image_name(image),
                                score=1.0,
                            ),
                        )
                    )
                decision = "folder_expanded"
                reason = "matched_folder"
                row_extra = {
                    "matched_folder": folder_result["matched_folder"],
                    "expanded_scene_count": len(folder_images),
                    "deduplicated_scene_count": added,
                }
                folder_expansions[entry] = {
                    "matched_folder": folder_result["matched_folder"],
                    "scene_count": len(folder_images),
                    "deduplicated_scene_count": added,
                    "preview": [_canonical_scene_entry(item) for item in folder_images[:5]],
                }
            elif folder_result["status"] == "ambiguous":
                decision = "ambiguous_folder"
                reason = "folder_basename_ambiguous"
                candidates = list(folder_result["candidates"])
                row_extra = {"folder_candidates": candidates}
                ambiguous.append(
                    {
                        "entry": entry,
                        "normalized_scene_line": needle,
                        "reason": "folder_basename_ambiguous",
                        "candidates": [{"folder": item} for item in candidates],
                    }
                )
            else:
                if best and best["score"] >= 0.85:
                    decision = "likely_missing_file"
                    reason = "best_candidate_below_accept_threshold"
                missing.append(entry)

        rows.append(
            {
                "original_scene_line": entry,
                "normalized_scene_line": needle,
                "scene_signature": scene_signature(needle),
                "exact_match": bool(best and best["score"] == 1.0),
                "best_candidate_1": _image_name(best["image"]) if best else None,
                "best_candidate_1_key": _image_key(best["image"]) if best else None,
                "best_candidate_1_normalized": best["normalized_image"] if best else None,
                "best_candidate_1_score": best["score"] if best else None,
                "best_candidate_1_reason": best["reason"] if best else None,
                "best_candidate_2": _image_name(second["image"]) if second else None,
                "best_candidate_2_key": _image_key(second["image"]) if second else None,
                "best_candidate_2_score": second["score"] if second else None,
                "best_candidate_2_reason": second["reason"] if second else None,
                "decision": decision,
                "reason": reason,
                **row_extra,
            }
        )

    return {
        "schema_version": 1,
        "total_scenes_in_scenes_txt": len(entries),
        "requested_entries_count": len(entries),
        "requested_files_count": len(requested_files),
        "requested_folders_count": len(requested_folders),
        "expanded_scene_count": len(matched),
        "requested_files": requested_files,
        "requested_folders": requested_folders,
        "folder_expansions": folder_expansions,
        "unresolved_entries": missing,
        "total_images_available": len(images),
        "total_tif_images_available": len([item for item in images if _is_raster_image(item)]),
        "available_tiff_samples": [_canonical_image_path(item) for item in images if _is_raster_image(item)][:20],
        "matched": [item.__dict__ for item in matched],
        "missing": missing,
        "ambiguous": ambiguous,
        "rows": rows,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "total_entries": len(entries),
        "total_images": len(images),
        "accept_threshold": accept_threshold,
        "ambiguous_margin": ambiguous_margin,
        "preferred_key_prefixes": preferred_key_prefixes or [],
    }


def match_scenes(entries: list[str], images: list[dict[str, Any]]) -> tuple[list[SceneMatch], list[dict[str, Any]], list[str]]:
    report = build_scene_matching_report(entries, images)
    return [SceneMatch(**item) for item in report["matched"]], report["ambiguous"], report["missing"]


def _append_match_once(matched: list[SceneMatch], seen: set[str], match: SceneMatch) -> bool:
    identity = _normalized_identity(match.key or match.entry or match.name)
    if identity in seen:
        return False
    seen.add(identity)
    matched.append(match)
    return True


def _normalized_identity(value: str) -> str:
    return _normalize_path(value).casefold()


def _normalize_path(value: str) -> str:
    return str(value or "").replace("\\", "/").strip().strip("/")


def _canonical_image_path(image: dict[str, Any]) -> str:
    if not isinstance(image, dict):
        return _normalize_path(str(image))
    key = _normalize_path(str(image.get("key") or ""))
    name = _normalize_path(str(image.get("name") or ""))
    return key or name


def _image_key(image: dict[str, Any]) -> str:
    if not isinstance(image, dict):
        return _normalize_path(str(image))
    return _normalize_path(str(image.get("key") or image.get("name") or ""))


def _image_name(image: dict[str, Any]) -> str:
    if not isinstance(image, dict):
        return PurePosixPath(_normalize_path(str(image))).name
    name = _normalize_path(str(image.get("name") or ""))
    return name or PurePosixPath(_image_key(image)).name


def _canonical_scene_entry(image: dict[str, Any]) -> str:
    return _canonical_image_path(image)


def _is_raster_image(image: dict[str, Any]) -> bool:
    path = _canonical_image_path(image)
    name = _image_name(image)
    return path.casefold().endswith(RASTER_SUFFIXES) or name.casefold().endswith(RASTER_SUFFIXES)


def _build_folder_index(images: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, set[str]] = {}
    for image in images:
        if not _is_raster_image(image):
            continue
        path = _canonical_image_path(image)
        if "/" not in path:
            continue
        parts = [part for part in path.split("/")[:-1] if part]
        for idx in range(1, len(parts) + 1):
            folder = "/".join(parts[:idx])
            norm = folder.casefold()
            bucket = by_path.setdefault(norm, {"folder": folder, "images": []})
            bucket["images"].append(image)
            basename = parts[idx - 1].casefold()
            by_basename.setdefault(basename, set()).add(norm)
    for bucket in by_path.values():
        bucket["images"].sort(key=lambda item: _canonical_image_path(item).casefold())
    return {"by_path": by_path, "by_basename": by_basename}


def _match_folder_entry(entry: str, folder_index: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_path(entry)
    if not normalized:
        return {"status": "missing"}
    normalized_key = normalized.casefold()
    by_path: dict[str, dict[str, Any]] = folder_index.get("by_path") or {}
    by_basename: dict[str, set[str]] = folder_index.get("by_basename") or {}

    candidates: list[str]
    if "/" in normalized_key:
        candidates = sorted(
            norm_path
            for norm_path in by_path
            if norm_path == normalized_key or norm_path.endswith("/" + normalized_key)
        )
    else:
        candidates = sorted(by_basename.get(normalized_key) or [])
    if not candidates:
        return {"status": "missing"}
    if len(candidates) > 1:
        return {"status": "ambiguous", "candidates": [by_path[item]["folder"] for item in candidates]}
    folder = by_path[candidates[0]]
    images = list(folder.get("images") or [])
    if not images:
        return {"status": "missing"}
    return {"status": "matched", "matched_folder": folder["folder"], "images": images}
