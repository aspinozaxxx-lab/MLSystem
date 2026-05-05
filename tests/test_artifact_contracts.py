from __future__ import annotations

import unittest

from mlsystem.src.contracts.artifacts import DatasetManifestArtifact, InferenceManifestArtifact, InventoryScenesArtifact


class ArtifactContractsTests(unittest.TestCase):
    def test_inventory_contract_accepts_existing_json_shape(self) -> None:
        artifact = InventoryScenesArtifact(matched_count=1, matched=[{"entry": "scene.tif", "key": "images/scene.tif"}])
        self.assertEqual(artifact.safe_dump()["matched_count"], 1)

    def test_dataset_manifest_contract_serializes(self) -> None:
        artifact = DatasetManifestArtifact(
            experiment_id="unit",
            split_strategy="object_balanced",
            object_count_mode="property",
            selected_scene_count=2,
            train_scene_count=1,
            val_scene_count=1,
        )
        self.assertEqual(artifact.safe_dump()["split_strategy"], "object_balanced")

    def test_inference_manifest_masks_secret_fields(self) -> None:
        artifact = InferenceManifestArtifact(
            experiment_id="unit",
            run_on="dataset_scenes",
            scenes=[{"entry": "scene.tif", "token": "secret"}],
        )
        self.assertEqual(artifact.safe_dump()["scenes"][0]["token"], "***")


if __name__ == "__main__":
    unittest.main()
