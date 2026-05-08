# Metrics Debug Report

Дата: 2026-05-08.

Базовый класс проверки: `deforest` / `cuttings` / «вырубки».

## Краткий вывод

Найден и исправлен явный источник недостоверных pixel F1/IoU: training loop публиковал validation metrics как невзвешенное среднее batch-level F1/IoU. Теперь source of truth для pixel metrics - единая confusion matrix по всем validation pixels на эпоху.

Исправлено:

- `val/pixel_f1`, `val/pixel_iou`, `val/precision`, `val/recall`, `val/pixel_accuracy` считаются из глобальных TP/FP/FN/TN.
- `val/pixel_tp`, `val/pixel_fp`, `val/pixel_fn`, `val/pixel_tn` и `val/threshold` пишутся в MLflow/history/training_result.
- `evaluate_pixel_metrics` и `compute_f1` берут counts/threshold из `training_result.json`, если они доступны.
- `train/loss_total`, `train/loss_bce`, `train/loss_dice`, `val/loss_total`, `val/loss_bce`, `val/loss_dice` логируются отдельно и усредняются по количеству samples, а не как last batch.
- Object matching tie-break стал deterministic: `(-iou, pred_index, gt_index)`.
- Добавлен metrics debug dump на каждую эпоху.

Оставшиеся риски:

- Object F1 по validation все еще зависит от наличия validation vectorization artifacts. Если distinct validation vectorization stage не запущен, `compute_f1` честно публикует object metrics как unavailable.
- Полный server debug run по «вырубкам» должен быть выполнен после деплоя этого кода через текущий CI/CD контур. Локально проверены unit/smoke guarantees, но не полноценная GPU-тренировка.

## Как включить debug mode

Через config:

```yaml
metrics_debug:
  enabled: true
  class_name: "вырубки"
  class_id: 1
  save_every_epoch: true
  save_all_val_samples: true
  save_arrays: true
  save_png: true
  save_object_matching: true
  upload_to_mlflow: true
```

Через env fallback:

```bash
MLSYSTEM_METRICS_DEBUG=1
MLSYSTEM_METRICS_DEBUG_CLASS=вырубки
```

Artifacts пишутся в:

```text
<experiment_dir>/metrics_debug/<run_id>/epoch_0001/
<experiment_dir>/metrics_debug/<run_id>/epoch_0002/
```

На каждую эпоху пишутся:

- `epoch_summary.json`
- `per_sample_metrics.csv`
- `val_manifest_snapshot.json`
- `metrics_recompute_check.json`
- `mlflow_logged_metrics.json`
- `samples/<sample_id>/metadata.json`
- `samples/<sample_id>/image.png`
- `samples/<sample_id>/gt_mask.png`
- `samples/<sample_id>/pred_prob.npz`
- `samples/<sample_id>/pred_prob.png`
- `samples/<sample_id>/pred_mask.png`
- `samples/<sample_id>/overlay_gt_pred.png`
- `samples/<sample_id>/confusion_map.png`
- `samples/<sample_id>/tp_mask.png`
- `samples/<sample_id>/fp_mask.png`
- `samples/<sample_id>/fn_mask.png`
- `samples/<sample_id>/objects_gt.geojson`
- `samples/<sample_id>/objects_pred.geojson`
- `samples/<sample_id>/object_matches.json`
- `samples/<sample_id>/object_iou_matrix.csv`

`metrics_recompute_check.json` пересчитывает global metrics из `per_sample_metrics.csv` и сверяет с тем, что ушло в MLflow. Допуск: `1e-6`.

## Проверка гипотез скачков

| Гипотеза | Статус | Вывод |
|---|---|---|
| Нефиксированная val выборка | не подтверждено кодом | Split использует seed или prepared manifest. Debug сохраняет `val_manifest_snapshot.json` для проверки между эпохами. |
| Shuffle на val | не подтверждено | Val batches идут через `make_index_batches(..., shuffle=False)`. |
| Augmentations на val | не подтверждено | `_apply_train_augmentations` вызывается только в train loop. |
| `model.eval()` на val отсутствует | не подтверждено | Val loop вызывает `model.eval()`. |
| `torch.no_grad()` отсутствует | не подтверждено | Val loop внутри `torch.no_grad()`. |
| Accumulator не reset | исправлено/закрыто | Accumulators создаются на каждую эпоху. |
| Per-batch F1 mean без весов | подтверждено | Исправлено на micro TP/FP/FN/TN. |
| Train/val loss как last batch | частично | Был `np.mean(batch_loss)`; теперь sample-weighted average. |
| Loss components не логируются | подтверждено | Теперь логируются BCE и Dice components. |
| Threshold меняется между эпохами | не подтверждено | Threshold фиксируется `metrics.threshold`, default `0.5`, логируется как `val/threshold`. |
| Metrics на logits вместо probabilities | закрыто | `probabilities_from_logits`: sigmoid для binary, softmax channel для multi-class. |
| GT resize bilinear | не выявлено в исправленном участке | Metrics module не делает silent resize и падает при shape mismatch. |
| Две функции логируют один key | риск остается | `train_model`, `evaluate_pixel_metrics`, `compute_f1` могут логировать сходные normalized keys, но теперь читают один `training_result`. |
| Object F1 нестабилен из-за matching order | исправлено частично | Tie-break deterministic, но object F1 все еще зависит от vectorization/postprocess. |
| Псевдоразметка подмешивается в GT eval | не подтверждено | Pixel train validation берет GT masks из annotation rasterize; object metrics postprocess зависит от selected `gt_shapes`. |

## Таблица метрик по эпохам

После server debug run заполнить из `metrics_debug/<run_id>/epoch_*/epoch_summary.json`.

| epoch | train_loss | val_loss | pixel_f1_cuttings | pixel_iou_cuttings | object_f1_cuttings | gt_pixels | pred_pixels | gt_objects | pred_objects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pending_server_run | - | - | - | - | - | - | - | - | - |

## Топ-10 плохих samples

После server debug run заполнить из `per_sample_metrics.csv`, сортировка `pixel_f1 asc`, затем `object_f1 asc`.

| sample_id | scene_id | pixel_f1 | object_f1 | gt_pixels | pred_pixels | FP | FN | debug path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pending_server_run | - | - | - | - | - | - | - | - |

## Топ-10 скачков между эпохами

После server debug run сравнить `per_sample_metrics.csv` между соседними эпохами.

| sample_id | scene_id | epoch_a | epoch_b | f1_delta | iou_delta | likely reason | debug path |
|---|---|---:|---:|---:|---:|---|---|
| pending_server_run | - | - | - | - | - | - | - |

## Проверка консистентности

Локальные тесты подтверждают:

- perfect/empty/half-overlap pixel formulas;
- ignore_index не входит в TP/FP/FN/TN;
- threshold `0.5` работает ожидаемо;
- shape mismatch не ресайзится молча;
- loss average sample-weighted;
- debug recompute из per-sample rows совпадает с logged row;
- object matching one-to-one и deterministic tie order.

Локальный debug smoke для класса «вырубки» записал sample artifacts вне repo во временный каталог:

```text
<system-temp>/mlsystem_metrics_debug_smoke_cuttings/metrics_debug_cuttings_local_smoke/epoch_0001
```

`metrics_recompute_check.json` для smoke: `ok=true`, `pixel_f1=0.5625`, `pixel_iou=0.391304347826087`.

Команды локальной проверки:

```text
python -m unittest discover -s tests
python -m unittest discover -s frontend/tests
python -m compileall -q mlsystem airflow frontend tests
```

## Рекомендуемая server команда

Запуск должен идти через текущий Airflow/API контур после деплоя кода. Для controlled run:

```bash
MLSYSTEM_METRICS_DEBUG=1 \
MLSYSTEM_METRICS_DEBUG_CLASS=вырубки \
airflow dags trigger mlsystem_experiment_pipeline \
  -r metrics_debug_cuttings_YYYYMMDD_HHMMSS \
  -c '<deforest/cuttings config with fixed seed, fixed val manifest, short epoch count>'
```

В отчете server run нужно сохранить:

- git commit hash;
- `run_id`;
- MLflow run id;
- путь `metrics_debug_root`;
- `metrics_recompute_check.json` по каждой эпохе;
- сравнение `training_result.last_epoch_metrics`, `pixel_metrics.json`, MLflow metrics.

## Source of truth

Для pixel quality использовать:

- `val/cuttings_pixel_f1`
- `val/cuttings_pixel_iou`
- `val/cuttings_pixel_precision`
- `val/cuttings_pixel_recall`
- `val/cuttings_gt_pixels`
- `val/cuttings_pred_pixels`

Для общих/legacy графиков сохраняются алиасы:

- `val/pixel_f1`
- `val/pixel_iou`
- `val/precision`
- `val/recall`
- `val/dice`
- `val/iou`

Эти значения теперь берутся из одной global confusion matrix.
