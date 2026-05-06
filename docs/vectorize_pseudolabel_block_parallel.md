# Block-parallel vectorize_pseudolabel

`vectorize_pseudolabel` поддерживает новый режим:

```json
{
  "pseudolabel": {
    "vectorization": {
      "mode": "block_parallel",
      "core_size_px": 4096,
      "halo_px": 512,
      "workers": 30,
      "memory_guard_enabled": true,
      "local_min_area": 0,
      "final_min_area": 100,
      "merge_epsilon": 1.0,
      "bad_block_policy": "fail"
    }
  }
}
```

Default пока остается `legacy`. Новый режим нужно включать явно.

## Схема

Нельзя делать `prediction tile -> vectorize tile -> final polygons`, потому что это дает артефакты на границах tile. Новый путь использует:

```text
probability maps -> spatial blocks with halo -> vectorize expanded block -> clip to core -> final merge/dissolve
```

## Core и halo

- Core block - область, за которую отвечает конкретный worker.
- Expanded block - core плюс `halo_px` со всех сторон.
- Векторизация выполняется на expanded block.
- В итог блока попадают только геометрии, clipped to core.
- Если polygon касается границы core или был clipped, он попадает в boundary candidates счетчик.

## Parallel execution

Используется `ProcessPoolExecutor`.

`workers_effective` считается как минимум из:

- `pseudolabel.vectorization.workers`;
- `cpu_count - 2`;
- memory guard, если задан `max_worker_memory_mb`.

Каждый worker пишет только свои файлы:

```text
vectorization_block_parallel/blocks/<block_id>.geojson
vectorization_block_parallel/blocks/<block_id>.summary.json
```

Записи из 30 процессов в один общий GeoJSON нет.

## Artifacts

```text
prediction_tiles_manifest.json
prediction_tiles_manifest.txt
vectorization_plan.json
vectorization_plan.txt
processing_blocks.geojson
block_tile_intersections.json
block_results.json
block_results.txt
failed_blocks.txt
vectorization_summary.json
vectorization_summary.txt
<experiment_id>.accepted.geojson
```

## XCom counters

Stage report пушит scalar counters:

```text
vectorization_mode
counter_prediction_tiles
counter_prediction_scenes
counter_blocks_total
counter_blocks_done
counter_blocks_failed
counter_workers_requested
counter_workers_effective
counter_boundary_candidates
counter_polygons_before_merge
counter_polygons_after_merge
counter_accepted_objects
counter_final_objects
counter_final_geojson_size_mb
metric_vectorization_duration_sec
metric_merge_duration_sec
metric_area_ratio_after_merge_to_before_merge
metric_area_ratio_final_to_before_merge
```

## Safety

- `all_images` с `block_parallel` запрещен без явного `pseudolabel.vectorization.allow_all_images=true`.
- `bad_block_policy=fail` является default.
- Старый legacy path остается доступен через `pseudolabel.vectorization.mode=legacy`.
- `local_min_area` применяется только внутри блока как технический фильтр.
- `final_min_area` применяется после final merge/dissolve.

## Ограничения MVP

- Сейчас входом является `pseudolabel_scene_results_manifest.json`, где один probability map соответствует сцене.
- Weighted blending helper есть, но текущий MVP обычно обрабатывает один scene-level probability map на block.
- CRS должен быть projected/metric. EPSG:4326-like CRS для block_parallel сейчас fail-fast.

## Как валидировать

1. Запустить малый run не на `all_images`.
2. Включить `pseudolabel.vectorization.mode=block_parallel`.
3. Для первой проверки поставить `workers=4`.
4. Проверить:
   - `blocks_failed=0`;
   - `blocks_done=blocks_total`;
   - `accepted.geojson` создан;
   - XCom values читаются без `Error loading XCom entry`.
5. После этого повторить с `workers=30`.
