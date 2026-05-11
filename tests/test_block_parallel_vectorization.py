from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from InferenceEngine.src.inference_engine.vectorization import run_block_parallel_vectorization
from InferenceEngine.src.inference_engine.vectorization.block_grid import build_processing_blocks
from InferenceEngine.src.inference_engine.vectorization.contracts import BlockVectorizationJob
from InferenceEngine.src.inference_engine.vectorization.tile_index import build_prediction_tile_index


class BlockParallelVectorizationTests(unittest.TestCase):
    def test_plan_blocks_create_core_and_halo_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_scene(Path(tmp))
            tiles = build_prediction_tile_index(manifest, Path(tmp) / "index")
            blocks = build_processing_blocks(tiles, core_size_px=8, halo_px=2)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].core_window, (0, 0, 8, 8))
        self.assertEqual(blocks[0].expanded_window, (0, 0, 10, 8))
        self.assertEqual(blocks[1].core_window, (8, 0, 8, 8))
        self.assertEqual(blocks[1].expanded_window, (6, 0, 10, 8))

    def test_block_parallel_merges_polygon_split_by_core_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_synthetic_scene(root)
            accepted = root / "accepted.geojson"
            summary = run_block_parallel_vectorization(
                run_id="unit",
                manifest_path=manifest,
                output_dir=root / "vectorization",
                accepted_geojson=accepted,
                threshold=0.5,
                class_name="deforest",
                core_size_px=8,
                halo_px=2,
                workers_requested=1,
                memory_guard_enabled=True,
                local_min_area=0,
                final_min_area=0,
                merge_epsilon=0,
            )
            accepted_payload = json.loads(accepted.read_text(encoding="utf-8"))
            self.assertTrue((root / "vectorization" / "blocks" / "000000_0000_0000.geojson").exists())
            self.assertTrue((root / "vectorization" / "blocks" / "000000_0000_0001.geojson").exists())
        self.assertEqual(summary["blocks_total"], 2)
        self.assertEqual(summary["blocks_failed"], 0)
        self.assertEqual(summary["final_objects"], 1)
        self.assertEqual(len(accepted_payload["features"]), 1)

    def test_block_job_contract_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_synthetic_scene(Path(tmp))
            tiles = build_prediction_tile_index(manifest, Path(tmp) / "index")
            block = build_processing_blocks(tiles, core_size_px=8, halo_px=2)[0]
            job = BlockVectorizationJob(run_id="unit", block=block, tiles=tiles, threshold=0.5, local_min_area=0, output_dir=tmp)
            encoded = json.dumps(job.to_dict(), ensure_ascii=False)
            decoded = BlockVectorizationJob.from_dict(json.loads(encoded))
        self.assertEqual(decoded.block.block_id, job.block.block_id)
        self.assertEqual(decoded.tiles[0].scene_id, "scene_a")


def _write_synthetic_scene(root: Path) -> Path:
    results_dir = root / "pseudolabel_scene_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prob = np.zeros((8, 16), dtype=np.uint8)
    prob[2:6, 6:11] = 255
    npz_path = results_dir / "scene_0000.npz"
    np.savez(npz_path, prob_uint8=prob)
    meta_path = results_dir / "scene_0000.json"
    meta = {
        "scene_index": 0,
        "scene_id": "scene_a",
        "scene_name": "scene_a.tif",
        "npz_path": npz_path.name,
        "transform": [1, 0, 0, 0, -1, 0],
        "crs": "EPSG:3857",
        "coverage_fraction": 1.0,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scene_count": 1,
        "scene_result_dir": str(results_dir),
        "scenes": [{"scene_index": 0, "scene_id": "scene_a", "scene_name": "scene_a.tif", "npz_path": str(npz_path), "meta_path": str(meta_path)}],
    }
    manifest_path = root / "pseudolabel_scene_results_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    unittest.main()
