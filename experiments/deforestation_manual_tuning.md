# Deforestation Manual Tuning

Manual Codex-operated tuning only: one Airflow DAG run, wait, verify MLflow, analyze, then decide the next single run. No tuning supervisors or background loops.

## Baseline before updated dataset run

- Class: `Вырубки` / `deforestation`
- Updated dataset source: `/data/MLMarkup/Вырубки`
- Runtime layout: `/data/mlsystem/tuning/deforestation/layout`
- Dataset stats: 355 objects, 24 scenes
- Baseline Airflow run: `train_all_deforestation_v2_20260512_223035`
- Baseline MLflow run: `dfc8b8d0cd404f2999d6515382adc7ba`
- Baseline checkpoint: `/data/mlsystem/airflow/status/train_all_deforestation_v2_20260512_223035/segformer_b2.pt`
- Baseline best `val/pixel_f1`: 0.5299933751244245 at epoch 26
- Baseline best precision/recall: 0.38286865907828316 / 0.8607553188613446
- Note: baseline was unstable near the end, with high recall and low precision.

## 2026-05-13 manual_deforestation_20260513_232042_warmbest_lr5e5

- Airflow run id: `manual_deforestation_20260513_232042_warmbest_lr5e5`
- MLflow run id: `035dc44a54204954ad5921e9a45af41e`
- Hypothesis: updated deforestation dataset plus warm-start from the best class checkpoint, with lower LR `5e-5`, should preserve learned masks and reduce instability/false positives without changing split/loss/architecture.
- Initial checkpoint: `/data/mlsystem/airflow/status/train_all_deforestation_v2_20260512_223035/segformer_b2.pt`
- Config summary: SegFormer B2, patch 1024, stride 512, batch 2, focal_dice, AdamW, weight_decay 1e-4, epochs 25, early stopping patience 8, object-balanced scene split seed 20260513.
- Pseudo-labeling: disabled.
- Airflow state: success
- MLflow verified: yes, `035dc44a54204954ad5921e9a45af41e`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/035dc44a54204954ad5921e9a45af41e`
- Dataset used: 355 source objects, 24 scenes; prepared split counted 322 matched objects, 20 train scenes, 4 val scenes.
- Prepared tiles: 24 train tiles, 7 val tiles.
- best `val/pixel_f1`: 0.41997880800372467 at epoch 7.
- best threshold-sweep F1: 0.44530329110635203 at threshold 0.7.
- last `val/pixel_f1`: 0.38819041059890713 at epoch 15.
- best precision/recall: 0.28985274401391675 / 0.7621269653210064.
- Output checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260513_232042_warmbest_lr5e5/segformer_b2.pt`
- Stability / overfit notes: worse than baseline and still noisy; the prepared dataset is too small for meaningful epoch dynamics, so the short 1.57s epochs are explained by only 24/7 train/val tiles.
- Conclusion: do not promote this checkpoint. Keep baseline checkpoint `train_all_deforestation_v2_20260512_223035/segformer_b2.pt`.
- Next experiment: expand tile sampling while holding architecture, loss, split seed, and LR fixed.

## 2026-05-13 manual_deforestation_20260513_235518_fulltiles_lr5e5

- Airflow run id: `manual_deforestation_20260513_235518_fulltiles_lr5e5`
- MLflow run id: `8ecf9bf71e654719b851e2b12deef211`
- Hypothesis: the previous run is dominated by micro-dataset sampling. Increasing train/val tile coverage should make epoch metrics less noisy and improve precision/F1 while keeping the same class checkpoint and LR.
- Initial checkpoint: `/data/mlsystem/airflow/status/train_all_deforestation_v2_20260512_223035/segformer_b2.pt`
- Config summary: SegFormer B2, patch 1024, stride 512, batch 2, focal_dice, AdamW, learning_rate 5e-5, weight_decay 1e-4, epochs 25, early stopping patience 8, object-balanced scene split seed 20260513, `use_full_dataset_tiles=true`, max train tiles 512, max val tiles 128, max tiles per scene 32.
- Pseudo-labeling: disabled.
- Airflow state: failed at `train_model`
- MLflow verified: partial failed run, `8ecf9bf71e654719b851e2b12deef211`, no metrics.
- Prepared tiles: 512 train tiles, 55 val tiles.
- Failure: CUDA OOM after caching samples on GPU, `cached_mb=11340.0`; model forward then failed while allocating 256 MiB.
- Conclusion: invalid run, do not compare for F1. The sampling change is still the right hypothesis, but GPU sample cache must be disabled for this tile count.
- Next experiment: rerun the same expanded sampling with `cache_samples_on_gpu=false`.

## 2026-05-14 manual_deforestation_20260514_000042_fulltiles_nocache

- Airflow run id: `manual_deforestation_20260514_000042_fulltiles_nocache`
- MLflow run id: `f0b8f5c089ef4ffabda9742ac76540d4`
- Hypothesis: expanded tile coverage is needed, but should be streamed from CPU/RAM instead of cached on GPU to avoid OOM. This should produce a valid metric with less sampling noise.
- Initial checkpoint: `/data/mlsystem/airflow/status/train_all_deforestation_v2_20260512_223035/segformer_b2.pt`
- Config summary: same as previous expanded-sampling run, except `cache_samples_on_gpu=false`.
- Pseudo-labeling: disabled.
- Airflow state: success
- MLflow verified: yes, `f0b8f5c089ef4ffabda9742ac76540d4`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/f0b8f5c089ef4ffabda9742ac76540d4`
- Prepared tiles: 512 train tiles, 55 val tiles.
- best `val/pixel_f1`: 0.5235649435926173 at epoch 18.
- best threshold-sweep F1: 0.525717723105703 at threshold 0.3.
- last `val/pixel_f1`: 0.28598155603391673 at epoch 25.
- best precision/recall: 0.568632795464457 / 0.4851163165208748.
- last precision/recall: 0.8583881173402235 / 0.17157120664594436.
- Output checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260514_000042_fulltiles_nocache/segformer_b2.pt`
- Stability / overfit notes: valid updated-dataset run, normal epoch duration around 33s, but recall collapses late. The best threshold is lower than 0.5, so the model is too conservative.
- Conclusion: promote this as current updated-dataset baseline, but tune recall/stability next.
- Next experiment: start from this checkpoint and add positive emphasis in focal_dice.

## 2026-05-14 manual_deforestation_20260514_002020_posweight_recall

- Airflow run id: `manual_deforestation_20260514_002020_posweight_recall`
- MLflow run id: `9d23e90adb15494bbb2987b5e24962fd`
- Hypothesis: current updated-dataset baseline is precision-heavy/recall-low; adding positive emphasis to focal_dice should raise recall at threshold 0.5 and improve pixel F1 without changing sampling, architecture, or split.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260514_000042_fulltiles_nocache/segformer_b2.pt`
- Config summary: SegFormer B2, same full-tile sampling, `cache_samples_on_gpu=false`, LR 5e-5, focal_dice with `pos_weight=2.0` and `focal_alpha=0.65`, epochs 18, patience 6.
- Pseudo-labeling: disabled.
- Airflow state: success
- MLflow verified: yes, `9d23e90adb15494bbb2987b5e24962fd`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/9d23e90adb15494bbb2987b5e24962fd`
- Prepared tiles: 512 train tiles, 55 val tiles.
- best `val/pixel_f1`: 0.5357019447550986 at epoch 1.
- best threshold-sweep F1: 0.5398891127811047 at threshold 0.3.
- last `val/pixel_f1`: 0.36878007466285384 at epoch 7.
- best precision/recall: 0.7165114648578069 / 0.42775839593291304.
- last precision/recall: 0.43893230713801124 / 0.31796190299020294.
- Output checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt`
- Stability / overfit notes: improved best F1, but best at epoch 1 and then rapid degradation. Positive emphasis at LR 5e-5 is too aggressive.
- Conclusion: promote this as current best checkpoint by F1, but next run must stabilize it with lower LR and softer positive weighting.
- Next experiment: lower LR to 1e-5 and reduce positive weighting.

## 2026-05-14 manual_deforestation_20260514_003009_softpos_lr1e5

- Airflow run id: `manual_deforestation_20260514_003009_softpos_lr1e5`
- MLflow run id: `e4462600547e4c789ec08cd82de11e16`
- Hypothesis: the positive-weight run found a better point but moved too fast; lower LR and softer positive emphasis should preserve the epoch-1 gain while reducing rapid degradation.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt`
- Config summary: same full-tile sampling, `cache_samples_on_gpu=false`, LR 1e-5, focal_dice with `pos_weight=1.5` and `focal_alpha=0.6`, epochs 12, patience 5.
- Pseudo-labeling: disabled.
- Airflow state: success
- MLflow verified: yes, `e4462600547e4c789ec08cd82de11e16`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/e4462600547e4c789ec08cd82de11e16`
- Prepared tiles: 512 train tiles, 55 val tiles.
- best `val/pixel_f1`: 0.4252841773230398 at epoch 1.
- best threshold-sweep F1: 0.4252841773230398 at threshold 0.5.
- last `val/pixel_f1`: 0.32139369008013546 at epoch 6.
- best precision/recall: 0.9039475733019207 / 0.2780497251912.
- last precision/recall: 0.8997021709915184 / 0.19564040376555386.
- Output checkpoint: `/data/mlsystem/airflow/status/manual_deforestation_20260514_003009_softpos_lr1e5/segformer_b2.pt`
- Stability / overfit notes: lower LR and softer positive weighting did not preserve the previous gain; the model remained too conservative and recall fell further.
- Conclusion: do not promote this checkpoint. Current best remains `manual_deforestation_20260514_002020_posweight_recall` by peak F1, while `manual_deforestation_20260514_000042_fulltiles_nocache` is the more stable updated-dataset baseline.
- Next experiment: if tuning resumes, do not continue the soft-positive branch; test threshold-aware/recall-oriented evaluation or sampling changes from the `manual_deforestation_20260514_002020_posweight_recall` checkpoint.
