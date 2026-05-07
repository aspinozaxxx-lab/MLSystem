from __future__ import annotations

import unittest

from mlsystem.src.data.scene_matching import build_scene_matching_report, norm_scene_name, parse_scene_list_text


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

    def test_preferred_prefix_resolves_exact_duplicates(self) -> None:
        images = [
            {"key": "images/kanopus/Oktyabrskij/scene_a.tif", "name": "scene_a.tif"},
            {"key": "images/kanopus/wave_2_Upload_01/scene_a.tif", "name": "scene_a.tif"},
        ]
        report = build_scene_matching_report(
            ["scene_a.tif"],
            images,
            preferred_key_prefixes=["images/kanopus/wave_2_Upload_01/"],
        )
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["ambiguous_count"], 0)
        self.assertEqual(report["matched"][0]["key"], "images/kanopus/wave_2_Upload_01/scene_a.tif")

    def test_missing_candidate(self) -> None:
        report = build_scene_matching_report(["missing.tif"], [{"key": "x/other.tif", "name": "other.tif"}])
        self.assertEqual(report["missing_count"], 1)

    def test_norm_scene_name_removes_cog_suffix(self) -> None:
        self.assertEqual(norm_scene_name("ABC_cog.tif"), "abc")

    def test_old_format_concrete_files_still_matches(self) -> None:
        images = [
            {"key": "Hilokskij/a.tif", "name": "a.tif"},
            {"key": "Toguchinskij/b.tif", "name": "b.tif"},
        ]
        report = build_scene_matching_report(["Hilokskij/a.tif", "Toguchinskij/b.tif"], images)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual([item["entry"] for item in report["matched"]], ["Hilokskij/a.tif", "Toguchinskij/b.tif"])
        self.assertEqual(report["requested_files_count"], 2)
        self.assertEqual(report["requested_folders_count"], 0)

    def test_folder_direct_match_expands_tif_scenes(self) -> None:
        images = [
            {"key": "Hilokskij/a.tif", "name": "a.tif"},
            {"key": "Hilokskij/b.tif", "name": "b.tif"},
            {"key": "Toguchinskij/c.tif", "name": "c.tif"},
        ]
        report = build_scene_matching_report(["Hilokskij"], images)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual([item["entry"] for item in report["matched"]], ["Hilokskij/a.tif", "Hilokskij/b.tif"])
        self.assertEqual(report["folder_expansions"]["Hilokskij"]["scene_count"], 2)
        self.assertEqual(report["requested_folders_count"], 1)

    def test_folder_match_is_case_insensitive(self) -> None:
        images = [
            {"key": "Irkutsk/a.tif", "name": "a.tif"},
            {"key": "Irkutsk/B.TIF", "name": "B.TIF"},
        ]
        report = build_scene_matching_report(["irkutsk"], images)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual([item["entry"] for item in report["matched"]], ["Irkutsk/a.tif", "Irkutsk/B.TIF"])
        self.assertEqual(report["folder_expansions"]["irkutsk"]["matched_folder"], "Irkutsk")

    def test_folder_trailing_slash_and_backslash(self) -> None:
        images = [
            {"key": "Hilokskij/a.tif", "name": "a.tif"},
            {"key": "Toguchinskij/c.tif", "name": "c.tif"},
        ]
        report = build_scene_matching_report(["Hilokskij/", r"Toguchinskij\\"], images)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual([item["entry"] for item in report["matched"]], ["Hilokskij/a.tif", "Toguchinskij/c.tif"])

    def test_mixed_file_and_folder_deduplicates(self) -> None:
        images = [
            {"key": "Hilokskij/a.tif", "name": "a.tif"},
            {"key": "Hilokskij/b.tif", "name": "b.tif"},
            {"key": "Toguchinskij/c.tif", "name": "c.tif"},
        ]
        report = build_scene_matching_report(["Hilokskij", "Hilokskij/a.tif", "Toguchinskij/c.tif"], images)
        self.assertEqual(report["matched_count"], 3)
        self.assertEqual([item["entry"] for item in report["matched"]], ["Hilokskij/a.tif", "Hilokskij/b.tif", "Toguchinskij/c.tif"])

    def test_unknown_folder_is_unresolved(self) -> None:
        report = build_scene_matching_report(["HILOKSKI_typo"], [{"key": "Hilokskij/a.tif", "name": "a.tif"}])
        self.assertEqual(report["matched_count"], 0)
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["unresolved_entries"], ["HILOKSKI_typo"])

    def test_ambiguous_folder_basename_requires_more_specific_path(self) -> None:
        images = [
            {"key": "region1/irkutsk/a.tif", "name": "a.tif"},
            {"key": "region2/irkutsk/b.tif", "name": "b.tif"},
        ]
        ambiguous = build_scene_matching_report(["irkutsk"], images)
        self.assertEqual(ambiguous["matched_count"], 0)
        self.assertEqual(ambiguous["ambiguous_count"], 1)
        self.assertIn("region1/irkutsk", str(ambiguous["ambiguous"]))
        self.assertIn("region2/irkutsk", str(ambiguous["ambiguous"]))

        specific = build_scene_matching_report(["region1/irkutsk"], images)
        self.assertEqual(specific["matched_count"], 1)
        self.assertEqual(specific["matched"][0]["entry"], "region1/irkutsk/a.tif")

    def test_parse_scene_list_text_handles_bom_crlf_comments(self) -> None:
        text = "\ufeff Hilokskij\r\n\r\n# comment\r\nToguchinskij\\\r\n"
        self.assertEqual(parse_scene_list_text(text), ["Hilokskij", "Toguchinskij/"])

    def test_folder_expansion_ignores_non_raster_files(self) -> None:
        images = [
            {"key": "Hilokskij/a.tif", "name": "a.tif"},
            {"key": "Hilokskij/readme.txt", "name": "readme.txt"},
        ]
        report = build_scene_matching_report(["Hilokskij"], images)
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["matched"][0]["entry"], "Hilokskij/a.tif")


if __name__ == "__main__":
    unittest.main()
