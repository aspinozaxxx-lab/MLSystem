from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass
class SceneMatch:
    entry: str
    key: str
    name: str
    score: float


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
) -> dict[str, Any]:
    normalized_images = [(item, norm_scene_name(item["name"])) for item in images]
    matched: list[SceneMatch] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[str] = []
    rows: list[dict[str, Any]] = []

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
        if best and best["score"] >= accept_threshold:
            if len(exact_candidates) == 1:
                decision = "matched"
                reason = exact_candidates[0]["reason"]
                image = exact_candidates[0]["image"]
                matched.append(SceneMatch(entry=entry, key=image["key"], name=image["name"], score=1.0))
            elif second and second["score"] >= accept_threshold and (best["score"] - second["score"]) <= ambiguous_margin:
                decision = "ambiguous"
                reason = "multiple_close_candidates"
                ambiguous.append(
                    {
                        "entry": entry,
                        "normalized_scene_line": needle,
                        "candidates": [
                            {
                                "name": item["image"]["name"],
                                "key": item["image"]["key"],
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
                matched.append(SceneMatch(entry=entry, key=image["key"], name=image["name"], score=round(float(best["score"]), 4)))
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
                "best_candidate_1": best["image"]["name"] if best else None,
                "best_candidate_1_key": best["image"]["key"] if best else None,
                "best_candidate_1_normalized": best["normalized_image"] if best else None,
                "best_candidate_1_score": best["score"] if best else None,
                "best_candidate_1_reason": best["reason"] if best else None,
                "best_candidate_2": second["image"]["name"] if second else None,
                "best_candidate_2_key": second["image"]["key"] if second else None,
                "best_candidate_2_score": second["score"] if second else None,
                "best_candidate_2_reason": second["reason"] if second else None,
                "decision": decision,
                "reason": reason,
            }
        )

    return {
        "schema_version": 1,
        "total_scenes_in_scenes_txt": len(entries),
        "total_images_available": len(images),
        "total_tif_images_available": len([item for item in images if item["name"].lower().endswith((".tif", ".tiff"))]),
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
    }


def match_scenes(entries: list[str], images: list[dict[str, Any]]) -> tuple[list[SceneMatch], list[dict[str, Any]], list[str]]:
    report = build_scene_matching_report(entries, images)
    return [SceneMatch(**item) for item in report["matched"]], report["ambiguous"], report["missing"]
