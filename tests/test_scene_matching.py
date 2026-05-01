from __future__ import annotations

import unittest

from mlsystem.src.data.scene_matching import build_scene_matching_report, norm_scene_name


class SceneMatchingTests(unittest.TestCase):
    def test_normalized_exact_match(self) -> None:
        images = [{"key": "images/a/SCENE_01_cog.tif", "name": "SCENE_01_cog.tif"}]
        report = build_scene_matching_report(["scene-01.tif"], images)
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["matched"][0]["key"], "images/a/SCENE_01_cog.tif")

    def test_kanopus_signature_match(self) -> None:
        images = [{"key": "x/KANOPUS_20240101112233_SCN02_cog.tif", "name": "KANOPUS_20240101112233_SCN02_cog.tif"}]
        report = build_scene_matching_report(["kanopus-20240101-112233-scn2.tif"], images)
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["rows"][0]["best_candidate_1_reason"], "kanopus_datetime_scn_signature")

    def test_ambiguous_candidates(self) -> None:
        images = [
            {"key": "a/scene_a.tif", "name": "scene_a.tif"},
            {"key": "b/scene_a_cog.tif", "name": "scene_a_cog.tif"},
        ]
        report = build_scene_matching_report(["scene_a.tif"], images)
        self.assertEqual(report["ambiguous_count"], 1)

    def test_unique_exact_candidate_wins_over_close_signature(self) -> None:
        images = [
            {
                "key": "a/KV3_35352_37872-01_KANOPUS_20240616_074631_7.L2.PMS.SCN01.tif",
                "name": "KV3_35352_37872-01_KANOPUS_20240616_074631_7.L2.PMS.SCN01.tif",
            },
            {
                "key": "a/KV3_35352_37872-01_KANOPUS_20240616_074631_7.L2.PMS.SCN05 (1).tif",
                "name": "KV3_35352_37872-01_KANOPUS_20240616_074631_7.L2.PMS.SCN05 (1).tif",
            },
        ]
        report = build_scene_matching_report(
            ["KV3_35352_37872-01_KANOPUS_20240616_074631_7.L2.PMS.SCN01"],
            images,
        )
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["ambiguous_count"], 0)
        self.assertEqual(report["matched"][0]["key"], images[0]["key"])

    def test_missing_candidate(self) -> None:
        report = build_scene_matching_report(["missing.tif"], [{"key": "x/other.tif", "name": "other.tif"}])
        self.assertEqual(report["missing_count"], 1)

    def test_norm_scene_name_removes_cog_suffix(self) -> None:
        self.assertEqual(norm_scene_name("ABC_cog.tif"), "abc")


if __name__ == "__main__":
    unittest.main()
