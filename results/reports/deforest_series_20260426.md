# Deforest Experiment Series Report

Generated: 2026-04-26

Source:
- Repo summaries: `results/summaries/*.json`
- Server verification: `/data/mlsystem/storage/jobs`, `/data/mlsystem/storage/experiments`
- MLflow experiment: <http://172.26.12.169:5000/#/experiments/7>

## Queue Status

| Job | Status | Notes |
| --- | --- | --- |
| `deforest_exp01_r18_t512` | interrupted/stale-running | Claimed and logged 1 epoch, then executor was restarted by code deploy. Queue dir remains in `running`; no final `run_summary.json`. |
| `deforest_exp02_r18_t768` | done | Completed with early stopping. |
| `deforest_exp03_r18_t1024` | done | Completed with early stopping. |
| `deforest_exp04_r34_t512` | done | Completed with early stopping. |
| `deforest_exp05_r34_t768` | interrupted/stale-running | Claimed, matched scenes, then executor was restarted by code deploy before epoch metrics. No final `run_summary.json`. |
| `deforest_exp06_r34_t1024` | done | Completed with early stopping. |
| `deforest_exp07_r50_t768` | done | Completed with early stopping. |
| `deforest_exp08_r50_t1024` | done | Completed with early stopping. |
| `deforest_exp09_tiny_t1024` | done | Completed with early stopping. |
| `deforest_exp10_r34_t1024_aug` | done | Completed with early stopping. |

## Metrics

| Job | Model | Tile | Duration min | Epochs | Best epoch | Best val IoU | Final val IoU | Final val Dice | Final val F1 | Accepted objects | GeoJSON MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deforest_exp01_r18_t512` | unet_resnet18 | 512 | 3.32* | 1* | n/a | n/a | 0.2513* | 0.2865* | 0.2865* | n/a | n/a |
| `deforest_exp02_r18_t768` | unet_resnet18 | 768 | 34.88 | 14 | 4 | 0.2310 | 0.1870 | 0.2525 | 0.2525 | 81 | 0.57 |
| `deforest_exp03_r18_t1024` | unet_resnet18 | 1024 | 26.27 | 12 | 2 | 0.3553 | 0.1443 | 0.2178 | 0.2178 | 159 | 0.89 |
| `deforest_exp04_r34_t512` | unet_resnet34 | 512 | 52.82 | 30 | 20 | 0.4484 | 0.2039 | 0.2748 | 0.2748 | 341 | 1.16 |
| `deforest_exp05_r34_t768` | unet_resnet34 | 768 | n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `deforest_exp06_r34_t1024` | unet_resnet34 | 1024 | 45.05 | 13 | 3 | 0.1841 | 0.1042 | 0.1462 | 0.3129 | 93 | 0.75 |
| `deforest_exp07_r50_t768` | unet_resnet50 | 768 | 76.11 | 17 | 9 | 0.1746 | 0.0979 | 0.1445 | 0.1445 | 99 | 0.96 |
| `deforest_exp08_r50_t1024` | unet_resnet50 | 1024 | 35.30 | 10 | 2 | 0.2047 | 0.0007 | 0.0014 | 0.0014 | 500 | 3.06 |
| `deforest_exp09_tiny_t1024` | tiny_unet_4ch | 1024 | 11.73 | 11 | 1 | 0.3333 | 0.2433 | 0.3137 | 0.4248 | 500 | 3.03 |
| `deforest_exp10_r34_t1024_aug` | unet_resnet34 | 1024 | 115.46 | 30 | 18 | 0.2001 | 0.1078 | 0.1487 | 0.5486 | 215 | 1.42 |

`*` = partial interrupted run, not a completed job summary.

## Findings

- Best completed run by `best_val_iou`: `deforest_exp04_r34_t512` (`0.4484`).
- Best completed run by final `val/iou`: `deforest_exp09_tiny_t1024` (`0.2433`), but it is the lightweight baseline.
- Best completed ResNet18 run: `deforest_exp03_r18_t1024` (`best_val_iou=0.3553`).
- ResNet50 did not justify the extra CPU cost in this run.
- All completed pseudolabel GeoJSON files are below the 20 MB limit.
- Every completed run has warning: CPU-only execution and 17 unmatched scenes from `scenes.txt`.

## Summary Format Gaps

Repo summaries before `67cdd1b` were missing:
- `started_at`
- `finished_at`
- `duration_sec`
- `epochs_completed` as a top-level field
- MLflow artifact pointers for pseudolabel outputs

`sync-results.yml` was updated in commit `67cdd1b` to include those fields on the next scheduled sync.
