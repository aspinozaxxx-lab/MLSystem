# Desertification Manual Tuning Journal

This is a human-readable journal for manually operated Airflow experiments on `Опустынивание` / `desertification`.
It is not an automation script and must not be used to launch or monitor experiments.

Policy:
- Launch one Airflow DAG run manually.
- Wait for completion.
- Verify MLflow run and metrics.
- Analyze pixel-level F1, precision, recall, stability, and checkpoint.
- Decide and launch the next experiment manually.
- Keep pseudo-labeling disabled.

## 2026-05-13 Manual takeover from deleted supervisor runs

- Airflow run id: `tune_desertification_airflow_20260512_234944_0007_9acfa1cb`
- MLflow run id: `0389ad44bb6440c5b57f511972643f6a`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/0389ad44bb6440c5b57f511972643f6a`
- Initial checkpoint: `/data/mlsystem/airflow/status/tune_desertification_airflow_20260512_233843_0002_8a913a65/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `5e-5`, AdamW, cosine scheduler, `focal_dice`, scene/object-balanced split seed `20260513`.
- Hypothesis: old supervisor attempted another stability follow-up from the early high-peak checkpoint.
- Airflow state: success, all tasks completed.
- MLflow verified: yes.
- best_val_pixel_f1: `0.5859725070251386`
- last_val_pixel_f1: `0.4791552225273993`
- precision: `0.8088224702916816`
- recall: `0.340408441632251`
- best_epoch: `3`
- output checkpoint: `/data/mlsystem/airflow/status/tune_desertification_airflow_20260512_234944_0007_9acfa1cb/segformer_b2.pt`
- Stability / overfit notes: weak run; precision is high but recall collapsed, and best epoch is very early.
- Conclusion: do not use this checkpoint as baseline.
- Next experiment: use the best observed checkpoint from run `0010` and target recall with a Tversky-biased loss at lower LR.

## Leaderboard snapshot after stopping supervisors

| Airflow run | MLflow run | best_val_pixel_f1 | last_val_pixel_f1 | precision | recall | best_epoch | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tune_desertification_airflow_20260512_235345_0010_a11de2f5` | `c3c6d51e918c49b9bf27014b28ac644f` | `0.6992568655` | `0.6254961163` | `0.7984700483` | `0.5141211685` | `2` | Best peak and best practical current baseline, but early peak and recall still low. |
| `tune_desertification_airflow_20260512_235820_0013_bc3fd086` | `0f81681e64b14d5b93bf4a2e068b2b5c` | `0.6977712961` | `0.4743724801` | `0.8715142672` | `0.3258742848` | `1` | High precision, recall collapse; do not continue from it. |
| `tune_desertification_airflow_20260512_233843_0002_8a913a65` | `8c6956871556460890db8d028924efac` | `0.6928577615` | `0.4104672367` | `0.7342422408` | `0.2848558330` | `5` | Strong peak but unstable; useful ancestor checkpoint only. |
| `tune_desertification_airflow_20260512_234624_0005_d77ef12e` | `5f137c242573463bb73f8ccb34e050de` | `0.6831234354` | `0.6193371689` | `0.8080516123` | `0.5020800970` | `2` | Stable-ish alternative; similar precision/recall profile to `0010`. |
| `train_all_desertification_v2_20260512_223035` | `c7463d7277094729b4bd2fd8a8c0e1d9` | `0.6314364513` | `0.6091473092` | `0.8336734022` | `0.4798999735` | `24` | Older stable baseline. |

## 2026-05-13 manual_desertification_20260513_095912_recall_tversky

- MLflow run id: `259bd2a5d22a4527affcdbf047ab1b9a`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/259bd2a5d22a4527affcdbf047ab1b9a`
- Hypothesis: use the best current checkpoint from run `0010`, improve recall with `focal_tversky` where `beta > alpha`, and reduce LR to avoid the early-epoch peak decay.
- Initial checkpoint: `/data/mlsystem/airflow/status/tune_desertification_airflow_20260512_235345_0010_a11de2f5/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1.2e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, epochs `45`, early stopping patience `8`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- best_val_pixel_f1: `0.7121840276911341`
- last_val_pixel_f1: `0.6722154856786315`
- precision: `0.6971122541997656`
- recall: `0.6490357291706135`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_095912_recall_tversky/segformer_b2.pt`
- Stability / overfit notes: clear improvement over the old best peak and much better last F1; still peaks very early, then drifts down. Epoch 2 was balanced (`precision=0.6724`, `recall=0.7570`); final epoch keeps a useful balance (`precision=0.6971`, `recall=0.6490`).
- Conclusion: this is the new baseline checkpoint.
- Next experiment: start from this checkpoint, keep the same loss/split/model, reduce LR again for a conservative refinement, and check whether the balanced early peak can be retained longer.

## 2026-05-13 manual_desertification_20260513_100528_low_lr_refine

- MLflow run id: `4e178d290eac4e988cc591805e8a93ce`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/4e178d290eac4e988cc591805e8a93ce`
- Hypothesis: refine the new best `focal_tversky` checkpoint with a lower LR to retain the balanced early peak and reduce post-peak drift.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_095912_recall_tversky/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `6e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, epochs `35`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- best_val_pixel_f1: `0.7098147338447696`
- last_val_pixel_f1: `0.6796243182713231`
- precision: `0.7153276725451775`
- recall: `0.6473155760997235`
- best_epoch: `3`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_100528_low_lr_refine/segformer_b2.pt`
- Stability / overfit notes: peak is slightly below the previous run, but final F1 is higher and the precision/recall balance is steadier. Lower LR helps stability but does not raise the best peak.
- Conclusion: keep `manual_desertification_20260513_095912_recall_tversky` as best checkpoint by `best_val_pixel_f1`; keep this run as evidence that conservative LR improves final stability.
- Next experiment: return to the best peak checkpoint and rebalance Tversky toward precision (`alpha=0.45,beta=0.55`) while keeping LR conservative, because recall is now high enough and precision is the likely limiter for the next F1 gain.

## 2026-05-13 manual_desertification_20260513_100948_precision_tversky

- MLflow run id: `7023b28262514d848d205eec03884089`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/7023b28262514d848d205eec03884089`
- Hypothesis: rebalance `focal_tversky` toward precision after recall improved; `alpha=0.45,beta=0.55` should reduce false positives while preserving enough recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_095912_recall_tversky/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `8e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.45,beta=0.55)`, epochs `35`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- best_val_pixel_f1: `0.6919430914952718`
- last_val_pixel_f1: `0.6468824525628175`
- precision: `0.7524196223077619`
- recall: `0.5673095138862577`
- best_epoch: `3`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_100948_precision_tversky/segformer_b2.pt`
- Stability / overfit notes: precision improved but recall fell too much; F1 is below both previous manual runs.
- Conclusion: do not use this checkpoint as baseline.
- Next experiment: address the suspiciously short epochs directly. Current runs use only `24` train tiles and `6` validation tiles, cached on GPU, so epoch time of `1.5-3s` is expected but the metric is noisy. Run a wider-tile experiment with more samples per scene before further LR/loss tuning.

## 2026-05-13 manual_desertification_20260513_101433_wider_tiles

- MLflow run id: `f76d77f1a06d4955b07ecd60750a9675`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/f76d77f1a06d4955b07ecd60750a9675`
- Hypothesis: widen tile sampling because prior epochs used only `24` train and `6` validation tiles; test whether F1 remains strong with broader scene coverage.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_095912_recall_tversky/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `6e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=64`, epochs `25`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- effective samples: `259` train tiles and `45` validation tiles (`32`/`227` train positive/negative, `5`/`40` val positive/negative).
- epoch duration: first epoch `25.28s`, later epochs about `16.1s`.
- best_val_pixel_f1: `0.5104740614260358`
- last_val_pixel_f1: `0.4087840152404189`
- precision: `0.3840772610613221`
- recall: `0.43688795249590195`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_101433_wider_tiles/segformer_b2.pt`
- Stability / overfit notes: wider validation sharply reduced measured F1; the prior `0.71` runs were likely over-optimistic because they validated on only `6` tiles.
- Conclusion: do not use this checkpoint as baseline. The next step is to improve the sampler before more tuning, because increasing the tile budget currently overfills with random empty tiles.
- Next experiment: after sampler fix/deploy, rerun a wider-tile Airflow experiment and judge against the wider-validation metric, not the previous 6-tile metric.
