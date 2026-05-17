from __future__ import annotations

import os
import unittest
from unittest import mock

from mlsystem.src.tile_preparation.dataloader import (
    DEFAULT_DATALOADER_WORKERS,
    DEFAULT_PREFETCH_FACTOR,
    resolve_prefetch_factor,
    resolve_worker_count,
)


class TilePreparationDataloaderDefaultTests(unittest.TestCase):
    def test_linux_default_workers_is_internal_sixteen(self) -> None:
        with mock.patch("platform.system", return_value="Linux"), _clean_env():
            self.assertEqual(resolve_worker_count(None), DEFAULT_DATALOADER_WORKERS)
            self.assertEqual(DEFAULT_DATALOADER_WORKERS, 16)
            self.assertEqual(resolve_prefetch_factor(DEFAULT_DATALOADER_WORKERS, None), DEFAULT_PREFETCH_FACTOR)

    def test_windows_default_workers_falls_back_to_zero(self) -> None:
        with mock.patch("platform.system", return_value="Windows"), _clean_env():
            self.assertEqual(resolve_worker_count(None), 0)
            self.assertIsNone(resolve_prefetch_factor(0, None))

    def test_env_override_changes_internal_default(self) -> None:
        with mock.patch("platform.system", return_value="Linux"), _clean_env():
            os.environ["MLSYSTEM_TILE_DATALOADER_WORKERS"] = "4"
            os.environ["MLSYSTEM_TILE_DATALOADER_PREFETCH_FACTOR"] = "3"
            self.assertEqual(resolve_worker_count(None), 4)
            self.assertEqual(resolve_prefetch_factor(4, None), 3)

    def test_explicit_worker_argument_still_works_for_debug_tools(self) -> None:
        with mock.patch("platform.system", return_value="Linux"), _clean_env():
            self.assertEqual(resolve_worker_count(2), 2)
            self.assertEqual(resolve_prefetch_factor(2, 5), 5)


class _clean_env:
    def __enter__(self) -> None:
        self._old_workers = os.environ.pop("MLSYSTEM_TILE_DATALOADER_WORKERS", None)
        self._old_prefetch = os.environ.pop("MLSYSTEM_TILE_DATALOADER_PREFETCH_FACTOR", None)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_workers is not None:
            os.environ["MLSYSTEM_TILE_DATALOADER_WORKERS"] = self._old_workers
        else:
            os.environ.pop("MLSYSTEM_TILE_DATALOADER_WORKERS", None)
        if self._old_prefetch is not None:
            os.environ["MLSYSTEM_TILE_DATALOADER_PREFETCH_FACTOR"] = self._old_prefetch
        else:
            os.environ.pop("MLSYSTEM_TILE_DATALOADER_PREFETCH_FACTOR", None)


if __name__ == "__main__":
    unittest.main()
