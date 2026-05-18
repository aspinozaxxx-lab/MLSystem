# metrics

## Назначение
`metrics` считает pixel/object metrics и пишет metric artifacts, используемые `train` и `train_pipeline`.

## Public API
- `PixelCounts`: DTO счетчиков `tp/fp/fn/tn`.
- `pixel_counts_from_masks(pred_mask, target_mask, ignore_index=None)`: считает pixel counters по маскам.
- `pixel_counts_from_logits(logits, target, threshold=0.5, class_index=1, ignore_index=None)`: считает pixel counters по logits.
- `metrics_from_counts(counts)`: возвращает precision/recall/F1/IoU/accuracy по counters.
- `binary_segmentation_metrics_from_logits(logits, target, threshold=0.5, class_index=1, ignore_index=None)`: возвращает pixel metrics по logits.
- `PixelMetricAccumulator(threshold=0.5, ignore_index=None)`: аккумулирует pixel metrics по батчам.
- `WeightedLossAccumulator()`: аккумулирует weighted losses по батчам.
- `compute_object_f1(pred_geoms, gt_geoms, iou_threshold=0.5)`: считает object-level precision/recall/F1.
- `compute_object_metrics_by_scene(pred_by_scene, gt_by_scene, iou_threshold=0.5)`: считает object metrics по сценам.
- `write_object_metrics_artifacts(experiment_dir, pred_features, gt_shapes, prefix="val", iou_threshold=0.5)`: пишет object metrics artifacts.

## Запрещенные пересечения
- Не запускает обучение и не владеет train loop.
- Не читает pipeline lifecycle/status.
- Не пишет в MLflow напрямую.
- Не делает vectorization/postprocess/pseudolabel export.
- Не содержит debug dump как production API.
