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

## 2026-05-15 no-Airflow pipeline runner tuning on fresh MLMarkup

From this point orchestration moved from Airflow to the MLSystem pipeline runner. Experiments were submitted one at a time through FastAPI `/api/v1/pipeline-runs`; no tuning supervisor or repeated launch script was used. Historical checkpoint paths still contain `/data/mlsystem/airflow/status/...`, but Airflow was not used to run these experiments.

### Environment and dataset

- Server API: `http://127.0.0.1:8088`, `/health` and `/ready` OK.
- Deployed MLSystem commit: `7ca87d62466c26d742f980074fc87fa71abfdef9`.
- Airflow containers: none observed.
- GPU after stop: RTX 5090 baseline memory about `15032 / 32607 MiB`.
- MLflow experiment for this stage: `mlsystem-cuttings-tuning`.
- Tuning workdir: `/data/mlsystem/tuning/cuttings`.
- Run root: `/data/mlsystem/runs`.
- Fresh MLMarkup repo: `/data/mlsystem/MLMarkup`.
- MLMarkup branch/commit: `main`, `1bbe6417d7ec646e16136828fa09987e7e3b81f4`.
- MLMarkup dirty: `false`.
- Class dir: `Вырубки`.
- Scenes file: `deforestation.txt`, 28 scenes.
- Annotation file: `deforestation.geojson`, 410 features.
- Snapshot artifact: `/data/mlsystem/tuning/cuttings/mlmarkup_snapshot.json`.

### Code fixes made before/while tuning

- `9b3a624` `Log best threshold metrics explicitly`: `real_train.py` now logs `val/best_pixel_f1`, `val/best_pixel_f1_threshold`, and related best-threshold precision/recall/IoU metrics.
- `7ca87d6` `Treat zombie pipeline workers as exited`: runner status refresh treats zombie worker PIDs as exited instead of leaving runs permanently `running`.
- Local test gate after fixes: `python -m unittest discover -s tests`, `python -m compileall -q mlsystem frontend tests monitoring scripts`, and `git diff --check` passed before deploy.

### Dataset diagnostic run

- Run id: `cuttings_dataset_diagnostic_20260515_0244_auto_objectbalanced_stable_20260514T234419Z_032f0eeb`.
- MLflow run id: `e2b97f0880c54e77bb98de42690ebb0f`.
- State: `succeeded`.
- Purpose: verify fresh MLMarkup split and tile preparation path before fine-tuning.
- Split: 23 train scenes, 5 validation scenes; no train/val overlap observed.
- Dataset: 410 objects total; 325 train objects, 85 validation objects.
- Tiles: 7620 train tiles, 2677 validation tiles.
- Positive tiles: 734 train, 98 validation.
- Diagnostic had explicit batch limits (`max_train_batches=3`, `max_val_batches=2`), so it is not leaderboard-eligible.
- Conclusion: dataset path is real `tile_preparation`; validation is separate fixed-grid data, not train leakage.

### Re-evaluation of old checkpoints on the fresh MLMarkup validation set

Primary metric is best threshold-sweep `val/pixel_f1` on the new validation set.

| rank | checkpoint | source MLflow run | re-eval run | new val/pixel_f1 | threshold | precision | recall | IoU | status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt` | `9d23e90adb15494bbb2987b5e24962fd` | `cuttings_reeval_20260515_0303_ckpt1_posweight_recall_20260515T000327Z_6a6f535e` | 0.6777148705502326 | 0.8 | 0.5727513724451786 | 0.8297823512028933 | 0.5125330803895866 | valid |
| 2 | `manual_deforestation_20260514_000042_fulltiles_nocache/segformer_b2.pt` | `f0b8f5c089ef4ffabda9742ac76540d4` | `cuttings_reeval_20260515_0335_ckpt2_fulltiles_nocache_retry1_20260515T003600Z_4c329abb` | 0.5440678140714351 | 0.8 | 0.4225799954285094 | 0.7635939825392181 | 0.3736903540767864 | valid |
| 3 | `train_all_deforestation_v2_20260512_223035/segformer_b2.pt` | `dfc8b8d0cd404f2999d6515382adc7ba` | `cuttings_reeval_20260515_0355_ckpt3_train_all_v2_20260515T005522Z_717784d8` | 0.35921943502715414 | 0.6 | 0.2831762182921332 | 0.49109702646769443 | 0.2189320392353008 | valid |

Best valid checkpoint on the new dataset remained `manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt`; its fresh validation result is stronger than all fine-tune attempts below.

### Fine-tune attempts from old checkpoints

All fine-tunes used MLSystem pipeline runner, `tile_preparation`, pseudo-labeling disabled, no batch limits, and MLflow logging. Because these were short sanity fine-tunes with low average GPU utilization, they are marked suspicious and not promoted over the checkpoint re-evaluation.

| rank | run id | MLflow run | parent | config | val/pixel_f1 | threshold | precision | recall | object_f1 | best epoch | median epoch sec | status |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `cuttings_ft_20260515_0748_ckpt1_lr1e5_bcedice_512_b8_20260515T054814Z_62e4e171` | `b97303dcad8b4f368696513f193b995c` | ckpt1 | SegFormer B2, 512/512, batch 8, aug 1, `bce_dice`, LR `1e-5` | 0.6532621866787768 | 0.8 | 0.5825931091601185 | 0.7434424346393644 | 0.2267379679144385 | 2 | 506.5972 | suspicious |
| 2 | `cuttings_ft_20260515_0649_ckpt1_lr1e5_bcedice_512_b4_20260515T043656Z_002ce02b` | `0ac0e246a7ef429aa60cf2b29d038d06` | ckpt1 | SegFormer B2, 512/512, batch 4, aug 1, `bce_dice`, LR `1e-5` | 0.5932204550189034 | 0.8 | 0.5120495980653332 | 0.7049739673226847 | 0.18181818181818182 | 2 | 513.2955 | suspicious |
| 3 | `cuttings_ft_20260515_0648_ckpt1_lr2e5_focaldice_512_b4_20260515T040211Z_4758e351` | `ef174cae4b764e66832b2dd09451ecaa` | ckpt1 | SegFormer B2, 512/512, batch 4, aug 1, `focal_dice`, LR `2e-5` | 0.545390769520011 | 0.8 | 0.41736661688603444 | 0.7867075472100152 | 0.15395003376097233 | 2 | 511.7649 | suspicious |
| 4 | `cuttings_ft_20260515_0650_ckpt2_lr2e5_focaldice_512_b4_20260515T051217Z_efc4ea1b` | `05bce091ee7d43a08e113a1a65e1ef05` | ckpt2 | SegFormer B2, 512/512, batch 4, aug 1, `focal_dice`, LR `2e-5` | 0.5228387491308193 | 0.8 | 0.3998179899892443 | 0.7552110876463818 | 0.1391193363114231 | 3 | 514.7740 | suspicious |

Batch 8 improved the best fine-tune result and used more VRAM, but average GPU utilization snapshots were still low, around 13.5%, with intermittent spikes up to about 59%. This points to an IO/data-loader bottleneck rather than a model-capacity issue.

### Invalid or manually stopped runs

- `cuttings_ft_20260515_0418_ckpt1_lr2e5_focaldice_1024_20260515T011847Z_cae1bd91`: invalid, worker exited before checkpoint load during earlier smoke interruption.
- `cuttings_ft_20260515_0430_ckpt1_lr2e5_focaldice_1024_retry1_20260515T013051Z_15735b19`: invalid, manually terminated after more than 87 minutes without first epoch metric.
- `cuttings_ft_20260515_0608_ckpt1_lr2e5_focaldice_768_sanity_20260515T031854Z_9e58510c`: invalid, manually terminated after 42m32s without first epoch metric.
- `cuttings_ft_20260515_0934_ckpt1_lr5e6_bcedice_512_b8_20260515T063338Z_61eedf91`: invalid, stopped by user request during `train_model` before first epoch metric. API cancel marker was set, then worker PID `5583` was terminated with SIGTERM inside `mlsystem-gpu-api`; GPU memory returned to baseline. Final API state became `failed` with `WorkerExited`, progress 18%.

### Current conclusion

- Best valid result on fresh MLMarkup: checkpoint re-evaluation of `manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt`, `val/pixel_f1=0.6777148705502326` at threshold `0.8`.
- Best fine-tune result: `cuttings_ft_20260515_0748_ckpt1_lr1e5_bcedice_512_b8_20260515T054814Z_62e4e171`, `val/pixel_f1=0.6532621866787768` at threshold `0.8`, but suspicious due short sanity length and low average GPU utilization.
- Do not promote any fine-tuned checkpoint yet.
- If tuning resumes, start from ckpt1 and address the loader/IO bottleneck before longer runs; otherwise the GPU spends too much time waiting between batches.
- The next scientifically useful run would be a controlled confirmation of ckpt1 without fine-tune drift, or a longer ckpt1 batch-8 fine-tune only after fixing resource utilization and keeping the early-best guard active.

### 2026-05-15 tile preparation bottleneck follow-up

- Action item opened from the suspicious fine-tune runs above: replace synchronous tile batch preparation with a facade DataLoader path, worker prefetch, pinned memory, STRtree geometry filtering, and `uint8_255` normalization fast path.
- Validation constraints remain unchanged: no augmentation, no random jitter, no virtual repeats, fixed validation grid.
- Benchmark target after deploy: compare old sync path, DataLoader workers 0/2/4/8, then run a short training smoke and record epoch duration, GPU utilization, data wait metrics, samples/sec, and MLflow run.
