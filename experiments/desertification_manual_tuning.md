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

## 2026-05-13 Cleanup Of Invalid Microdataset Runs

Old desertification experiments were removed because they used too few tiles per epoch and produced unreliable F1 estimates.

- MLflow runs deleted from active view: `30`
- Airflow status directories deleted: `18`
- Runtime log/API job directories deleted: `415`
- MLflow artifact directories deleted: `29`
- Cleanup manifests:
  - `/data/mlsystem/reports/desertification_experiment_cleanup/deleted_status_dirs_20260513_102812.json`
  - `/data/mlsystem/reports/desertification_experiment_cleanup/deleted_runtime_artifacts_20260513_102918.json`
  - `/data/mlsystem/reports/desertification_experiment_cleanup/deleted_mlflow_artifacts_20260513_102958.json`

Dataset/layout directories were kept.

## 2026-05-13 manual_desertification_20260513_103138_fixed_sampler_baseline

- MLflow run id: `702d7dde41514e41bb258c9fc8b8e30a`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/702d7dde41514e41bb258c9fc8b8e30a`
- Hypothesis: restart desertification after deleting invalid microdataset experiments. Use fixed positive sampler with wider tiles and a non-desertification deforestation warm-start checkpoint.
- Initial checkpoint: `/data/mlsystem/airflow/status/train_all_deforestation_v2_20260512_223035/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `3e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `25`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- epoch duration: first epoch `25.03s`, later epochs about `16-29s`.
- best_val_pixel_f1: `0.6789765064833625`
- last_val_pixel_f1: `0.6243518434218857`
- precision at last epoch: `0.5682799857134414`
- recall at last epoch: `0.6927001686636421`
- best_epoch: `15`
- best epoch precision/recall: `0.6759603460023934` / `0.6820197040477708`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_103138_fixed_sampler_baseline/segformer_b2.pt`
- Stability / overfit notes: this run has normal epoch duration and F1 progression: F1 grows from `0.395` at epoch 1 to `0.679` at epoch 15, then degrades to `0.624`.
- Conclusion: new valid baseline. Continue from this checkpoint.
- Next experiment: reduce false positives while keeping the same wide sampler by shifting Tversky toward precision.

## 2026-05-13 manual_desertification_20260513_104519_precision_balance

- MLflow run id: `e52604b7cf6647918a707529c443bfaf`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/e52604b7cf6647918a707529c443bfaf`
- Hypothesis: continue from the first valid wide-sampler baseline and shift `focal_tversky` toward precision to reduce false positives while preserving recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_103138_fixed_sampler_baseline/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.55,beta=0.45)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `20`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6783197170572297`
- last_val_pixel_f1: `0.6703046831632987`
- precision at last epoch: `0.7454443671193063`
- recall at last epoch: `0.6089258616018505`
- best_epoch: `5`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_104519_precision_balance/segformer_b2.pt`
- Stability / overfit notes: peak is slightly below baseline but final F1 is much more stable, and precision improves strongly.
- Conclusion: use this as the stable checkpoint for the next experiment, while keeping the previous baseline as best peak by a very small margin.
- Next experiment: test patch `512` to increase positive tile diversity; current patch `1024` still yields only `34` positive train tiles.

## 2026-05-13 manual_desertification_20260513_105401_patch512

- MLflow run id: `8e8fa67eda124c54b837c625bd3b24fa`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/8e8fa67eda124c54b837c625bd3b24fa`
- Hypothesis: test patch `512` to increase positive tile diversity while keeping the stable precision-oriented checkpoint and loss.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_104519_precision_balance/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `512`, stride `384`, batch `4`, LR `1e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.55,beta=0.45)`, `max_tiles_per_scene=64`, `max_train_tiles=576`, `max_val_tiles=192`, epochs `25`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1` history; `best_val_pixel_f1` is reconstructed from `training_result.json`.
- effective samples: `456` train tiles and `77` validation tiles (`72`/`384` train positive/negative, `11`/`66` val positive/negative).
- best_val_pixel_f1: `0.5202034493729812`
- last_val_pixel_f1: `0.4873894888887044`
- precision at last epoch: `0.5933758075079033`
- recall at last epoch: `0.41352701374000206`
- best_epoch: `6`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_105401_patch512/segformer_b2.pt`
- Stability / overfit notes: positive tile count increased, but recall collapsed and F1 stayed far below the valid `1024` patch baseline. This is not a usable branch.
- Conclusion: do not continue from this checkpoint. Return to patch `1024`; improve positive coverage by sampling more windows from the two positive training scenes while capping total negative tiles.
- Next experiment: use checkpoint `manual_desertification_20260513_104519_precision_balance/segformer_b2.pt`, patch `1024`, `max_tiles_per_scene=64`, `max_train_tiles=256`, `max_val_tiles=96`, and lower `max_empty_tile_share` to `0.25`.

## 2026-05-13 manual_desertification_20260513_110120_posbalanced1024

- MLflow run id: `9494ec9fb0364438b9b68fb9297c81bf`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/9494ec9fb0364438b9b68fb9297c81bf`
- Hypothesis: return to patch `1024` and rebalance sampling: more windows from positive training scenes, capped negative tiles, stable precision checkpoint.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_104519_precision_balance/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.55,beta=0.45)`, `max_tiles_per_scene=64`, `max_train_tiles=256`, `max_val_tiles=96`, `max_empty_tile_share=0.25`, epochs `25`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1` history; best step is epoch `1`.
- effective samples: `256` train tiles and `86` validation tiles (`98`/`158` train positive/negative, `19`/`67` val positive/negative).
- best_val_pixel_f1: `0.6245667739903543`
- last_val_pixel_f1: `0.6064015156810327`
- precision at last epoch: `0.6839690847387201`
- recall at last epoch: `0.5446354914065794`
- best_epoch: `1`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_110120_posbalanced1024/segformer_b2.pt`
- Stability / overfit notes: larger positive coverage did not improve F1. Best epoch was immediate and below the stable checkpoint, so this branch likely changed the training/validation tile distribution too much and reduced useful negative context.
- Conclusion: do not continue from this checkpoint. Keep current best peak `manual_desertification_20260513_103138_fixed_sampler_baseline` and stable checkpoint `manual_desertification_20260513_104519_precision_balance`.
- Next experiment: keep the original validated `1024` sampler and stable checkpoint; change only Tversky balance from precision-heavy `alpha=0.55,beta=0.45` to milder recall support `alpha=0.45,beta=0.55`.

## 2026-05-13 manual_desertification_20260513_110842_loss045055

- MLflow run id: `cc4996f3aa7f4a5ca8726ee230c28cfa`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/cc4996f3aa7f4a5ca8726ee230c28cfa`
- Hypothesis: keep the validated `1024` sampler and stable checkpoint; change only Tversky balance from precision-heavy to mild recall support.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_104519_precision_balance/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1e-5`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.45,beta=0.55)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `22`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6808542577389554`
- last_val_pixel_f1: `0.6450981039836163`
- precision at last epoch: `0.6823663416670889`
- recall at last epoch: `0.6116899407876295`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_110842_loss045055/segformer_b2.pt`
- Stability / overfit notes: this is the new best peak, but it is not yet stable; best F1 arrives very early and final F1 drops by about `0.036`.
- Conclusion: use this as the current best peak checkpoint, but the next step must stabilize it, not broaden the search.
- Next experiment: continue from this checkpoint with the same sampler/loss and lower LR to `5e-6`.

## 2026-05-13 manual_desertification_20260513_111618_lr5e6_stabilize

- MLflow run id: `7415452fe6714541959fb3906a40598f`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/7415452fe6714541959fb3906a40598f`
- Hypothesis: continue from the new best peak checkpoint with the same `1024` sampler and loss; lower LR to `5e-6` to stabilize early overfit.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_110842_loss045055/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `5e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.45,beta=0.55)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.682191048364867`
- last_val_pixel_f1: `0.6648869005564376`
- precision at last epoch: `0.7469228606461993`
- recall at last epoch: `0.5990879321712195`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_111618_lr5e6_stabilize/segformer_b2.pt`
- Stability / overfit notes: new best peak and better last F1 than the previous recall-balanced run. Precision is high while recall remains the limiting side.
- Conclusion: this is the current best peak checkpoint. Continue from it, but target recall carefully.
- Next experiment: keep LR `5e-6` and sampler fixed; shift Tversky slightly more toward recall with `alpha=0.40,beta=0.60`.

## 2026-05-13 manual_desertification_20260513_112359_loss040060

- MLflow run id: `f1f84c4c01aa4eba92ca06a71b62e8ba`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/f1f84c4c01aa4eba92ca06a71b62e8ba`
- Hypothesis: continue from current best checkpoint; keep LR and sampler fixed, shift Tversky to `alpha=0.40,beta=0.60` to improve recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_111618_lr5e6_stabilize/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `5e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6883975226304428`
- last_val_pixel_f1: `0.655936323872638`
- precision at last epoch: `0.6789368897707734`
- recall at last epoch: `0.6344430889407078`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6923370552995582`, but the comparable objective remains `val/pixel_f1` at the configured threshold.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_112359_loss040060/segformer_b2.pt`
- Stability / overfit notes: clear new best peak, with recall improved versus precision-heavy runs. Still peaks early and drifts down, so next change should reduce update size rather than alter data or architecture.
- Conclusion: current best checkpoint.
- Next experiment: continue from this checkpoint with the same sampler/loss and lower LR to `2e-6` to test whether the peak can stabilize or move later.

## 2026-05-13 manual_desertification_20260513_113131_lr2e6_stabilize

- MLflow run id: `b933809a96c74b0693e64cd701adadd9`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/b933809a96c74b0693e64cd701adadd9`
- Hypothesis: continue from current best checkpoint; keep sampler and loss fixed, lower LR to `2e-6` to stabilize the early F1 peak.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_112359_loss040060/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6894539458405688`
- last_val_pixel_f1: `0.672646800574474`
- precision at last epoch: `0.7364056879171813`
- recall at last epoch: `0.6190488122289753`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6964534497595629`, but the comparable objective remains `val/pixel_f1`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113131_lr2e6_stabilize/segformer_b2.pt`
- Stability / overfit notes: new best peak and best final F1 so far. Lower LR reduced the post-peak drop.
- Conclusion: current best checkpoint.
- Next experiment: continue from this checkpoint with the same sampler/loss and lower LR one more step to `1e-6`; if this stalls, stop lowering LR and test threshold/postprocess or regularization next.

## 2026-05-13 manual_desertification_20260513_113900_lr1e6_stabilize

- MLflow run id: `e6a8f20b59ee4312bca8af4b700d8da5`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/e6a8f20b59ee4312bca8af4b700d8da5`
- Hypothesis: continue from current best checkpoint; keep sampler and loss fixed, lower LR to `1e-6` to test whether the F1 peak stabilizes further.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113131_lr2e6_stabilize/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6883488424509773`
- last_val_pixel_f1: `0.6754326630154737`
- precision at last epoch: `0.7276596898874372`
- recall at last epoch: `0.6302006580544328`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6957653203603663`, but the comparable objective remains `val/pixel_f1`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113900_lr1e6_stabilize/segformer_b2.pt`
- Stability / overfit notes: lower LR no longer improves peak, but gives the best final F1 so far. Further LR reduction is not the next lever.
- Conclusion: keep `manual_desertification_20260513_113131_lr2e6_stabilize` as best peak, and keep this run as a stable alternative checkpoint.
- Next experiment: return to the best peak checkpoint and shift Tversky further toward recall with `alpha=0.35,beta=0.65`; keep LR `2e-6`.

## 2026-05-13 manual_desertification_20260513_114651_loss035065

- MLflow run id: `a771d88c5c32413a92038d400884d0e7`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/a771d88c5c32413a92038d400884d0e7`
- Hypothesis: return to the best peak checkpoint; keep LR/sampler fixed, shift Tversky further to `alpha=0.35,beta=0.65` to improve recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113131_lr2e6_stabilize/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.686762460113811`
- last_val_pixel_f1: `0.6704604772636147`
- precision at last epoch: `0.7236494480210776`
- recall at last epoch: `0.6245550458931108`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6948606735204279`, but the comparable objective remains `val/pixel_f1`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_114651_loss035065/segformer_b2.pt`
- Stability / overfit notes: recall-oriented shift beyond `alpha=0.40,beta=0.60` lowered peak and did not improve final F1.
- Conclusion: do not continue from this branch. Keep `alpha=0.40,beta=0.60` as the current best loss balance.
- Next experiment: return to best peak checkpoint, keep LR/loss/sampler fixed, and raise weight decay from `1e-4` to `3e-4` to test stability/false-positive control.

## 2026-05-13 manual_desertification_20260513_115430_wd3e4

- MLflow run id: `9699286d65334353bd5e16ca016bf618`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/9699286d65334353bd5e16ca016bf618`
- Hypothesis: return to best peak checkpoint; keep sampler/loss/LR fixed, raise weight decay to `3e-4` to test stability and false-positive control.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113131_lr2e6_stabilize/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, weight decay `3e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `18`, early stopping patience `6`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6889422240210462`
- last_val_pixel_f1: `0.6711138270209736`
- precision at last epoch: `0.736222474868358`
- recall at last epoch: `0.6165854147449922`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6939511017103851`, but the comparable objective remains `val/pixel_f1`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_115430_wd3e4/segformer_b2.pt`
- Stability / overfit notes: higher weight decay did not improve peak or final F1.
- Conclusion: do not continue from this branch. Keep weight decay `1e-4`.
- Next experiment: run threshold calibration from the best checkpoint with threshold `0.7`, because all recent threshold sweeps show best threshold near `0.7`.

## 2026-05-13 manual_desertification_20260513_120226_threshold07

- MLflow run id: `8ac7bbfe3b624967818e99a4c02db05c`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/8ac7bbfe3b624967818e99a4c02db05c`
- Hypothesis: threshold calibration. Recent sweeps peak near `0.7`; keep best checkpoint and training hyperparameters, set evaluation/postprocess threshold to `0.7`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_113131_lr2e6_stabilize/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `12`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6939810923314434`
- last_val_pixel_f1: `0.6856447218477316`
- precision at last epoch: `0.7549342229116621`
- recall at last epoch: `0.6280050552111115`
- best_epoch: `2`
- threshold note: this is a threshold-tuned result at `0.7`; it should be compared as deploy/postprocess F1, while the best threshold `0.5` comparable objective remains `0.6894539458405688` from `manual_desertification_20260513_113131_lr2e6_stabilize`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Stability / overfit notes: threshold `0.7` improves both peak and final F1 under the configured decision threshold. It is now the best deploy-style run.
- Conclusion: keep this as current deploy-style best checkpoint/config.
- Next experiment: continue from this threshold-tuned checkpoint and lower LR to `1e-6` while keeping threshold `0.7` to test whether final F1 stabilizes further.

## 2026-05-13 manual_desertification_20260513_120948_threshold07_lr1e6

- MLflow run id: `c9ddb8b2bc814cef989868bb45c44f33`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/c9ddb8b2bc814cef989868bb45c44f33`
- Hypothesis: continue from threshold-`0.7` deploy-style best checkpoint; keep threshold and loss fixed, lower LR to `1e-6` to stabilize final F1.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `1e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `12`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6934052741710484`
- last_val_pixel_f1: `0.6854652890219687`
- precision at last epoch: `0.756176582826096`
- recall at last epoch: `0.6268477438693036`
- best_epoch: `2`
- threshold note: threshold-tuned result at `0.7`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120948_threshold07_lr1e6/segformer_b2.pt`
- Stability / overfit notes: LR `1e-6` did not improve peak or final F1 versus threshold07 LR `2e-6`.
- Conclusion: keep `manual_desertification_20260513_120226_threshold07` as the deploy-style best.
- Next experiment: test intermediate threshold `0.65` from the deploy-style best checkpoint; include thresholds `0.55,0.60,0.65,0.70,0.75` in the sweep.

## 2026-05-13 manual_desertification_20260513_121713_threshold065

- MLflow run id: `45ffb9737b1040bda9987feaf2e7286a`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/45ffb9737b1040bda9987feaf2e7286a`
- Hypothesis: threshold calibration between `0.6` and `0.7`; keep deploy-style best checkpoint and training hyperparameters, evaluate/postprocess threshold `0.65` with expanded sweep.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.65`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow contains per-epoch `val/pixel_f1`; best step is epoch `2`.
- effective samples: `258` train tiles and `45` validation tiles (`34`/`224` train positive/negative, `8`/`37` val positive/negative).
- best_val_pixel_f1: `0.6929661282682519`
- last_val_pixel_f1: `0.6845241135983599`
- precision at last epoch: `0.7500256541816316`
- recall at last epoch: `0.6295444828822847`
- best_epoch: `2`
- threshold sweep note: within this run, threshold `0.75` had the best epoch-2 sweep F1 `0.6931982437908121`, but the configured `0.65` objective was below the current threshold-`0.7` best.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_121713_threshold065/segformer_b2.pt`
- Stability / overfit notes: threshold `0.65` is worse than `0.7`; the sweep suggests `0.75` is worth one explicit deploy-style check.
- Conclusion: keep `manual_desertification_20260513_120226_threshold07` as deploy-style best.
- Next queued experiments: `threshold075`, `threshold07_no_noise`, and `threshold07_batch4`.

## 2026-05-13 manual_desertification_20260513_122600_threshold075

- MLflow run id: `cee7b576a6564be0a9eaceb1c399ee9c`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/cee7b576a6564be0a9eaceb1c399ee9c`
- Hypothesis: explicitly set deploy/evaluation threshold to `0.75` because the previous sweep suggested it may be competitive.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6931970705072111`
- last_val_pixel_f1: `0.6839167610467981`
- precision at last epoch: `0.763398897168808`
- recall at last epoch: `0.6194246643562201`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122600_threshold075/segformer_b2.pt`
- Stability / overfit notes: threshold `0.75` increases precision but loses recall and does not beat threshold `0.7`.
- Conclusion: do not continue from this branch.
- Next experiment: keep threshold `0.7` branch and test stability levers.

## 2026-05-13 manual_desertification_20260513_122601_threshold07_no_noise

- MLflow run id: `380882f3a8f44802abc6b5bef7035328`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/380882f3a8f44802abc6b5bef7035328`
- Hypothesis: disable noise augmentation while keeping threshold `0.7`; if noise causes drift, final F1 and recall should improve.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `2`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6926353044494554`
- last_val_pixel_f1: `0.6898784088616648`
- precision at last epoch: `0.737529075207871`
- recall at last epoch: `0.6480113507342428`
- best_epoch: `1`
- threshold sweep note: best threshold-sweep F1 reached `0.694220007774244`, but configured threshold `0.7` remains below the current deploy-style best.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122601_threshold07_no_noise/segformer_b2.pt`
- Stability / overfit notes: peak is lower, but final F1 is the strongest stable final value so far and recall improves.
- Conclusion: no-noise is worth combining with the new batch4 branch, but it is not the peak baseline.
- Next experiment: test no-noise with batch size `4`.

## 2026-05-13 manual_desertification_20260513_122602_threshold07_batch4

- MLflow run id: `1f516090167c4ac4818de5057431b4e7`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/1f516090167c4ac4818de5057431b4e7`
- Hypothesis: increase batch size from `2` to `4` to reduce gradient noise while keeping threshold `0.7` and the same loss/sampler.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_120226_threshold07/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6951911734092013`
- last_val_pixel_f1: `0.6897119462674659`
- precision at last epoch: `0.7214752990875585`
- recall at last epoch: `0.6606274538054245`
- best_epoch: `2`
- threshold sweep note: best threshold-sweep F1 reached `0.6955162296285807`; MLflow reports best threshold `0.75`.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122602_threshold07_batch4/segformer_b2.pt`
- Stability / overfit notes: this is the best peak F1 so far and has strong final F1 with improved recall. It becomes the current baseline checkpoint.
- Conclusion: continue from this checkpoint.
- Next queued experiments: `batch4_threshold075`, `batch4_no_noise`, and `batch4_lr1e6`.

## 2026-05-13 manual_desertification_20260513_123742_batch4_threshold075

- MLflow run id: `a3722a9d508442b38a90c8034b375b60`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/a3722a9d508442b38a90c8034b375b60`
- Hypothesis: continue from the new batch4 best checkpoint and make the configured deploy threshold match the sweep-preferred `0.75`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122602_threshold07_batch4/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6918439107286516`
- last_val_pixel_f1: `0.6880029294337995`
- precision at last epoch: `0.7267061807893068`
- recall at last epoch: `0.653213770595522`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123742_batch4_threshold075/segformer_b2.pt`
- Stability / overfit notes: explicit threshold `0.75` loses peak F1 versus the batch4 threshold `0.7` baseline, despite acceptable final F1.
- Conclusion: do not continue from this branch. Keep `manual_desertification_20260513_122602_threshold07_batch4` as current baseline.
- Next experiment: wait for `batch4_no_noise` and `batch4_lr1e6`; threshold `0.7` remains the deploy setting for now.

## 2026-05-13 manual_desertification_20260513_123743_batch4_no_noise

- MLflow run id: `076e0ce6da8c48ec874a22a7819c320b`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/076e0ce6da8c48ec874a22a7819c320b`
- Hypothesis: combine the batch4 branch with no-noise augmentation because the earlier no-noise run had the best final F1 and better recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122602_threshold07_batch4/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6946465947313232`
- last_val_pixel_f1: `0.691683285810822`
- precision at last epoch: `0.7299651598247381`
- recall at last epoch: `0.6572165957506785`
- best_epoch: `4`
- threshold sweep note: threshold `0.75` reached `0.695512199905429`, slightly above the current configured-threshold best.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123743_batch4_no_noise/segformer_b2.pt`
- Stability / overfit notes: peak is just below current baseline, but this is the strongest final F1 and best stability so far; best epoch moved later to `4`.
- Conclusion: use this as the stable branch and test threshold `0.75` explicitly.
- Next queued experiments: `no_noise_threshold075`, `no_noise_lr1e6`, and `no_noise_threshold075_lr1e6`.

## 2026-05-13 manual_desertification_20260513_123744_batch4_lr1e6

- MLflow run id: `c26e1771ad654212aef458c735b30e7e`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/c26e1771ad654212aef458c735b30e7e`
- Hypothesis: continue from the batch4 best checkpoint and reduce LR to `1e-6` to slow early drift.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_122602_threshold07_batch4/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `1e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6936652427377554`
- last_val_pixel_f1: `0.689710344558727`
- precision at last epoch: `0.7163826308783261`
- recall at last epoch: `0.6649528853697994`
- best_epoch: `1`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123744_batch4_lr1e6/segformer_b2.pt`
- Stability / overfit notes: lower LR improves recall but lowers peak and does not beat no-noise stability.
- Conclusion: do not continue from this branch as-is. If using lower LR, combine it with no-noise only as a stability check.
- Next experiment: no-noise lower LR branch is already queued.

## 2026-05-13 manual_desertification_20260513_124650_no_noise_threshold075

- MLflow run id: `9676101acac640a1a55f439015afacca`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/9676101acac640a1a55f439015afacca`
- Hypothesis: continue from stable no-noise checkpoint and set threshold `0.75`, matching the branch's sweep-preferred threshold.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123743_batch4_no_noise/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6967686478727488`
- last_val_pixel_f1: `0.6939675255381933`
- precision at last epoch: `0.7432964446249032`
- recall at last epoch: `0.6507785620210822`
- best_epoch: `4`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_124650_no_noise_threshold075/segformer_b2.pt`
- Stability / overfit notes: this is the best peak and best final F1 so far; best epoch is later than most earlier branches.
- Conclusion: this becomes the current best checkpoint.
- Next queued experiments: lower LR on no-noise branch, threshold075+lower LR, and recall-oriented loss/sampling from this checkpoint.

## 2026-05-13 manual_desertification_20260513_124651_no_noise_lr1e6

- MLflow run id: `b7485c16c33a4d6ca7f2a1f7720be982`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/b7485c16c33a4d6ca7f2a1f7720be982`
- Hypothesis: continue from stable no-noise checkpoint with threshold `0.7` and reduce LR to `1e-6` to test stability.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123743_batch4_no_noise/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `1e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.7`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6949446285587466`
- last_val_pixel_f1: `0.6932221363303007`
- precision at last epoch: `0.733362456860775`
- recall at last epoch: `0.6572479167612822`
- best_epoch: `4`
- threshold sweep note: threshold `0.75` reached `0.6957814316558711`, but still below the current best configured threshold run.
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_124651_no_noise_lr1e6/segformer_b2.pt`
- Stability / overfit notes: final F1 is strong, but lower LR reduces peak versus threshold `0.75` at LR `2e-6`.
- Conclusion: keep as stable reference only; do not make it the baseline.
- Next experiment: wait for threshold `0.75` + LR `1e-6` branch before deciding whether lower LR is useful.

## 2026-05-13 manual_desertification_20260513_124652_no_noise_threshold075_lr1e6

- MLflow run id: `2e0dea694eba4cf7a848e26deafc643b`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/2e0dea694eba4cf7a848e26deafc643b`
- Hypothesis: combine no-noise, threshold `0.75`, and lower LR `1e-6` to keep the threshold gain while reducing later drift.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_123743_batch4_no_noise/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `1e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6957831650469151`
- last_val_pixel_f1: `0.6942365344425824`
- precision at last epoch: `0.7431949236891529`
- recall at last epoch: `0.6513298118077078`
- best_epoch: `7`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_124652_no_noise_threshold075_lr1e6/segformer_b2.pt`
- Stability / overfit notes: lower LR gives the strongest final F1 so far and a later best epoch, but peak is below the LR `2e-6` threshold `0.75` branch.
- Conclusion: keep as stable fallback, but peak baseline remains the `loss035065` branch.
- Next experiment: continue recall-oriented loss from the new best checkpoint.

## 2026-05-13 manual_desertification_20260513_125850_newbest_loss035065

- MLflow run id: `d4d95f8a19e8443895882eb48b6db9da`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/d4d95f8a19e8443895882eb48b6db9da`
- Hypothesis: continue from current best no-noise threshold `0.75` checkpoint; precision is higher than recall, so shift Tversky to `alpha=0.35,beta=0.65`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_124650_no_noise_threshold075/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, threshold `0.75`, noise augmentation disabled, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6969170205531963`
- last_val_pixel_f1: `0.6939899778759402`
- precision at last epoch: `0.7422395370657762`
- recall at last epoch: `0.6516304935095035`
- best_epoch: `4`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_125850_newbest_loss035065/segformer_b2.pt`
- Stability / overfit notes: small but real improvement over the prior best, with similar final F1. Recall-oriented loss helps slightly without destabilizing.
- Conclusion: this becomes the current best checkpoint.
- Next queued experiments: stronger recall loss `alpha=0.30,beta=0.70`, lower LR from this checkpoint, and lower empty-tile share.

## 2026-05-13 manual_desertification_20260513_130440_newbest_empty035

- MLflow run id: `0243b8cae06943d4a74ef043ded82a4c`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/0243b8cae06943d4a74ef043ded82a4c`
- Hypothesis: continue from current best no-noise threshold `0.75` checkpoint and reduce `max_empty_tile_share` from `0.50` to `0.35` to increase positive pressure and recall.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_124650_no_noise_threshold075/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.35`, sweep `[0.55,0.60,0.65,0.70,0.75]`, `max_tiles_per_scene=32`, `max_train_tiles=288`, `max_val_tiles=96`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.7187221426655912`
- last_val_pixel_f1: `0.7158508883808011`
- precision at last epoch: `0.7495691083489566`
- recall at last epoch: `0.6850356091770865`
- best_epoch: `1`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_130440_newbest_empty035/segformer_b2.pt`
- Stability / overfit notes: this is a large improvement, not a marginal threshold artifact. Recall rises substantially while precision remains strong; final F1 remains close to peak.
- Conclusion: this becomes the current best checkpoint and the sampling lever is the main active line.
- Next queued experiments: test `max_empty_tile_share=0.30` from this checkpoint while other recall/lower-LR branches finish.

## 2026-05-13 manual_desertification_20260513_131110_newbest_loss030070

- MLflow run id: `242d7f98e6a1423c8982a9d7511ec2d7`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/242d7f98e6a1423c8982a9d7511ec2d7`
- Hypothesis: continue from `loss035065` and push Tversky further toward recall with `alpha=0.30,beta=0.70`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_125850_newbest_loss035065/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.30,beta=0.70)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.50`, sweep `[0.55,0.60,0.65,0.70,0.75]`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.6964061584111856`
- last_val_pixel_f1: `0.6930299792178851`
- precision at last epoch: `0.7375479807440815`
- recall at last epoch: `0.6535802264195857`
- best_epoch: `4`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_131110_newbest_loss030070/segformer_b2.pt`
- Stability / overfit notes: stronger recall loss does not improve over `alpha=0.35,beta=0.65`, and both are far below the sampling-improved branch.
- Conclusion: do not continue from this branch; sampling is the stronger lever.
- Next experiment: continue sampling sweep.

## 2026-05-13 manual_desertification_20260513_131111_newbest_loss035065_lr1e6

- MLflow run id: `3a4a300aa41c476fa21d0972c811f611`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/3a4a300aa41c476fa21d0972c811f611`
- Hypothesis: continue from `loss035065` and lower LR to `1e-6` to stabilize the small gain.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_125850_newbest_loss035065/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `1e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.50`, sweep `[0.55,0.60,0.65,0.70,0.75]`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.696066768234333`
- last_val_pixel_f1: `0.6939110152546711`
- precision at last epoch: `0.7448815192265207`
- recall at last epoch: `0.6494693437778464`
- best_epoch: `1`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_131111_newbest_loss035065_lr1e6/segformer_b2.pt`
- Stability / overfit notes: lower LR does not beat the LR `2e-6` loss branch and is far below the sampling branch.
- Conclusion: do not continue from this branch.
- Next experiment: continue sampling sweep around `max_empty_tile_share=0.30-0.35`.

## 2026-05-13 manual_desertification_20260513_131620_empty030

- MLflow run id: `a7c487e5245948dca42cb1e203e32fda`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/a7c487e5245948dca42cb1e203e32fda`
- Hypothesis: continue from the sampling best checkpoint and reduce `max_empty_tile_share` from `0.35` to `0.30`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_130440_newbest_empty035/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.30`, sweep `[0.55,0.60,0.65,0.70,0.75]`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED` and contains per-epoch `val/pixel_f1`.
- best_val_pixel_f1: `0.7333752474301697`
- last_val_pixel_f1: `0.727866647451602`
- precision at last epoch: `0.7346218376754969`
- recall at last epoch: `0.7212345594907765`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_131620_empty030/segformer_b2.pt`
- Stability / overfit notes: large improvement over `0.35`; precision and recall are now much better balanced.
- Conclusion: this becomes the current best checkpoint and current best sampling value.
- Next queued experiments: bracket with `0.25` and `0.28`, and check lower LR at `0.30`.

## 2026-05-13 manual_desertification_20260513_132250_empty035_loss035065

- MLflow run id: `4494c21a06c14c5e97f0597e09358f5b`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/4494c21a06c14c5e97f0597e09358f5b`
- Hypothesis: continue from `empty035` and change only Tversky to `alpha=0.35,beta=0.65`.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_130440_newbest_empty035/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.35,beta=0.65)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.35`, sweep `[0.55,0.60,0.65,0.70,0.75]`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: success.
- MLflow verified: yes. MLflow run status is `FINISHED`.
- best_val_pixel_f1: `0.7181224241090697`
- last_val_pixel_f1: `0.7144876399746461`
- precision at last epoch: `0.7438912255385048`
- recall at last epoch: `0.6873201248760182`
- best_epoch: `1`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_132250_empty035_loss035065/segformer_b2.pt`
- Stability / overfit notes: loss shift no longer helps once sampling is corrected.
- Conclusion: do not continue from this branch.
- Next experiment: continue sampling sweep.

## 2026-05-13 manual_desertification_20260513_132850_empty025

- MLflow run id: `843d57920ab8410db728e0de4dc54bcb`
- MLflow URL: `http://31.192.104.147/mlflow/#/experiments/38/runs/843d57920ab8410db728e0de4dc54bcb`
- Hypothesis: continue from `empty030` and reduce `max_empty_tile_share` to `0.25`; test whether more positive pressure helps or starts creating false positives.
- Initial checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_131620_empty030/segformer_b2.pt`
- Config summary: `segformer_b2`, patch `1024`, stride `512`, batch `4`, LR `2e-6`, weight decay `1e-4`, AdamW, cosine scheduler, loss `focal_tversky(alpha=0.40,beta=0.60)`, threshold `0.75`, noise augmentation disabled, `max_empty_tile_share=0.25`, sweep `[0.55,0.60,0.65,0.70,0.75]`, epochs `10`, early stopping patience `5`, split seed `20260513`.
- Airflow state: post-train finalization still running when recorded; MLflow already verified.
- MLflow verified: yes. MLflow run status is `FINISHED`.
- best_val_pixel_f1: `0.7303236857130309`
- last_val_pixel_f1: `0.7242618837211839`
- precision at last epoch: `0.7237701000247707`
- recall at last epoch: `0.7247543361814702`
- best_epoch: `2`
- output checkpoint: `/data/mlsystem/airflow/status/manual_desertification_20260513_132850_empty025/segformer_b2.pt`
- Stability / overfit notes: recall remains strong, but precision drops enough that F1 is below `empty030`.
- Conclusion: `0.25` is too aggressive. Keep `max_empty_tile_share=0.30` as the current best.
- Next experiment: test a narrow `0.28` bracket and lower LR at `0.30`.
