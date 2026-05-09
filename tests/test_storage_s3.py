from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlsystem.src.storage.s3 import cached_s3_object_path


class StorageS3Tests(unittest.TestCase):
    def test_cached_s3_object_uses_unique_temporary_path(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.download_paths: list[str] = []

            def download_file(self, bucket: str, key: str, filename: str) -> None:
                self.download_paths.append(filename)
                Path(filename).write_bytes(b"raster")

        with tempfile.TemporaryDirectory() as tmp:
            old_cache = os.environ.get("MLSYSTEM_S3_CACHE_DIR")
            os.environ["MLSYSTEM_S3_CACHE_DIR"] = tmp
            client = FakeClient()
            try:
                with patch("mlsystem.src.storage.s3.s3_client", return_value=client):
                    config = SimpleNamespace(known_data_roots=[])
                    cached = cached_s3_object_path(config, "mlsystems", "images/a.tif")
            finally:
                if old_cache is None:
                    os.environ.pop("MLSYSTEM_S3_CACHE_DIR", None)
                else:
                    os.environ["MLSYSTEM_S3_CACHE_DIR"] = old_cache

            self.assertEqual(cached.read_bytes(), b"raster")
            self.assertEqual(len(client.download_paths), 1)
            tmp_name = Path(client.download_paths[0]).name
            self.assertRegex(tmp_name, r"^a\.tif\.\d+\.[0-9a-f]+\.tmp$")
            self.assertFalse(list((Path(tmp) / "s3" / "mlsystems" / "images").glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
