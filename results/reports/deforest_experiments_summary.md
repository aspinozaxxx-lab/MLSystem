# Deforest Experiments Summary

Generated: 2026-04-26

MLflow experiment: <http://172.26.12.169:5000/#/experiments/7>

## Overall Status

- Total jobs: 10
- Done: 8
- Failed: 0
- Stale/running: 2
- Missing repo summaries: 1
- Completed compute time: 6.63 h
- Completed wall-clock span: 6.65 h

## Results

| Job | Model | Tile | Status | Duration min | Epochs | Best IoU | Best Dice | Best F1 | Final Val IoU | Pseudo objects | GeoJSON MB |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deforest_exp01_r18_t512` | unet_resnet18 | 512 | interrupted_stale_running | n/a | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| `deforest_exp02_r18_t768` | unet_resnet18 | 768 | done | 34.88 | 14 | 0.2310 | 0.2887 | 0.4136 | 0.1870 | 81 | 0.57 |
| `deforest_exp03_r18_t1024` | unet_resnet18 | 1024 | done | 26.27 | 12 | 0.3553 | 0.4130 | 0.4130 | 0.1443 | 159 | 0.89 |
| `deforest_exp04_r34_t512` | unet_resnet34 | 512 | done | 52.82 | 30 | 0.4484 | 0.4521 | 0.4521 | 0.2039 | 341 | 1.16 |
| `deforest_exp05_r34_t768` | unet_resnet34 | 768 | interrupted_stale_running | n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| `deforest_exp06_r34_t1024` | unet_resnet34 | 1024 | done | 45.05 | 13 | 0.1841 | 0.2328 | 0.3992 | 0.1042 | 93 | 0.75 |
| `deforest_exp07_r50_t768` | unet_resnet50 | 768 | done | 76.11 | 17 | 0.1746 | 0.2168 | 0.2168 | 0.0979 | 99 | 0.96 |
| `deforest_exp08_r50_t1024` | unet_resnet50 | 1024 | done | 35.30 | 10 | 0.2047 | 0.2629 | 0.2629 | 0.0007 | 500 | 3.06 |
| `deforest_exp09_tiny_t1024` | tiny_unet_4ch | 1024 | done | 11.73 | 11 | 0.3333 | 0.3333 | 0.4248 | 0.2433 | 500 | 3.03 |
| `deforest_exp10_r34_t1024_aug` | unet_resnet34 | 1024 | done | 115.46 | 30 | 0.2001 | 0.2003 | 0.5758 | 0.1078 | 215 | 1.42 |

## Rankings

### Best Val IoU
1. `deforest_exp04_r34_t512`: 0.4484
2. `deforest_exp03_r18_t1024`: 0.3553
3. `deforest_exp09_tiny_t1024`: 0.3333
4. `deforest_exp02_r18_t768`: 0.2310
5. `deforest_exp08_r50_t1024`: 0.2047

### Best Val Dice
1. `deforest_exp04_r34_t512`: 0.4521
2. `deforest_exp03_r18_t1024`: 0.4130
3. `deforest_exp09_tiny_t1024`: 0.3333
4. `deforest_exp02_r18_t768`: 0.2887
5. `deforest_exp08_r50_t1024`: 0.2629

### Best Val F1
1. `deforest_exp10_r34_t1024_aug`: 0.5758
2. `deforest_exp04_r34_t512`: 0.4521
3. `deforest_exp09_tiny_t1024`: 0.4248
4. `deforest_exp02_r18_t768`: 0.4136
5. `deforest_exp03_r18_t1024`: 0.4130

## Pseudolabel Outputs

Completed jobs produced `accepted.geojson`, `accepted.geojson.gz`, `accepted.gpkg`, `pseudolabel_summary.json` in MLflow artifacts. Direct S3 artifact URIs are not present in current summaries; MLflow artifact paths are recorded instead.

| Job | Status | Objects | GeoJSON MB | <=20 MB | Postprocess |
| --- | --- | ---: | ---: | --- | --- |
| `deforest_exp01_r18_t512` | not_completed | n/a | n/a | n/a | n/a |
| `deforest_exp02_r18_t768` | done | 81 | 0.57 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp03_r18_t1024` | done | 159 | 0.89 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp04_r34_t512` | done | 341 | 1.16 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp05_r34_t768` | not_completed | n/a | n/a | n/a | n/a |
| `deforest_exp06_r34_t1024` | done | 93 | 0.75 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp07_r50_t768` | done | 99 | 0.96 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp08_r50_t1024` | done | 500 | 3.06 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp09_tiny_t1024` | done | 500 | 3.03 | yes | threshold=0.35, min_area=500, simplify=2 |
| `deforest_exp10_r34_t1024_aug` | done | 215 | 1.42 | yes | threshold=0.35, min_area=500, simplify=2 |

## Repo Summary Gaps

Current repo summaries before the updated `sync-results.yml` are missing timing and artifact URI fields for several jobs. The workflow now copies more lightweight files and normalizes `started_at`, `finished_at`, `duration_sec`, `epochs_completed`, and MLflow artifact pointers on future sync runs.

## Findings

- Best completed experiment by best validation IoU: `deforest_exp04_r34_t512`.
- Best compact/fast baseline: `deforest_exp09_tiny_t1024`.
- `deforest_exp08_r50_t1024` is weak despite producing many pseudolabel objects.
- ResNet50 is not worth the CPU cost in this series.
- `deforest_exp01_r18_t512` and `deforest_exp05_r34_t768` are stale/interrupted and need cleanup or retry.
- Every completed job reports only 7 matched scenes and 17 missing scene entries; data matching should be fixed before judging model quality.

## Links

- `deforest_exp01_r18_t512`: <http://172.26.12.169:5000/#/experiments/7/runs/48004d5c8eae4861b692eee980157469>
- `deforest_exp02_r18_t768`: <http://172.26.12.169:5000/#/experiments/7/runs/13aa48f80a6549e99c019385a98eed72>
- `deforest_exp03_r18_t1024`: <http://172.26.12.169:5000/#/experiments/7/runs/f62913c07bbe49c49e2459d2af413d64>
- `deforest_exp04_r34_t512`: <http://172.26.12.169:5000/#/experiments/7/runs/0a03fba3712247cbb5ffcbd2d240f655>
- `deforest_exp05_r34_t768`: <http://172.26.12.169:5000/#/experiments/7/runs/d10532a147264f3db54ed9b3db16eea2>
- `deforest_exp06_r34_t1024`: <http://172.26.12.169:5000/#/experiments/7/runs/3d146f622cda4c5286594e88e10f55a9>
- `deforest_exp07_r50_t768`: <http://172.26.12.169:5000/#/experiments/7/runs/3ed07cec48b44a2fa3d22b7fc9bf2dfd>
- `deforest_exp08_r50_t1024`: <http://172.26.12.169:5000/#/experiments/7/runs/3c34a178eb6f4d86867f5564d72896cd>
- `deforest_exp09_tiny_t1024`: <http://172.26.12.169:5000/#/experiments/7/runs/385bb58159da4d49a43e2634124ea8a4>
- `deforest_exp10_r34_t1024_aug`: <http://172.26.12.169:5000/#/experiments/7/runs/0e90b62da0ca48b6b2fcce954722ffc0>
