# Virtual Tile Sampling Debug

Virtual tile sampling is an opt-in training mode for small segmentation datasets. It expands the effective train epoch with lightweight tile records, repeat factors, denser positive windows, hard negatives, and optional train-only jitter. It does not save augmented image or mask tiles to disk.

## Why This Exists

The regular `prepare_dataset` stage keeps the split at scene level. Training then builds tiles later in `real_train.py`. On small datasets the epoch length can be too close to the number of base train tiles, even when online augmentation is enabled. Virtual tile sampling increases the train sampling space while preserving scene-level train/validation separation.

Validation stays honest:

- no online augmentation;
- no random jitter;
- no virtual repeats;
- fixed validation grid from `preprocess.stride`.

## Enable It

New behavior is disabled by default.

```yaml
preprocess:
  tile_size: 768
  stride: 512
  train_sampling:
    enabled: true
    virtual_epoch_multiplier: 1
    positive_repeat_factor: 4
    hard_negative_repeat_factor: 2
    negative_repeat_factor: 1
    positive_stride_factor: 0.5
    hard_negative_stride_factor: 0.5
    negative_stride_factor: 1.0
    min_positive_pixels: 1
    include_partial_positive: true
    max_empty_tile_share: 0.35
    batch_positive_fraction: 0.5
    batch_hard_negative_fraction: 0.25
    batch_negative_fraction: 0.25
    random_jitter:
      enabled: false
      max_shift_fraction: 0.25
      keep_positive_min_pixels: 1

train:
  augmentations:
    flips: true
    rot90: true
    brightness_contrast: true
    gamma: true
    noise: true
    blur: true
```

`positive_stride_factor: 0.5` means positive windows are sampled twice as densely per axis. `0.25` means four times as densely per axis. Dense stride is train-only.

## Supported Parameters

| YAML path | Default | Type | Allowed values / constraints | Scope | Affects | Count impact example |
|---|---:|---|---|---|---|---|
| `preprocess.train_sampling.enabled` | `false` | bool | `true` / `false` | train | window generation, repeats, balancing, jitter | `false` keeps the legacy `real_train._read_samples` path; `true` switches train to virtual records. |
| `preprocess.train_sampling.virtual_epoch_multiplier` | `1` | int | `>= 1` | train | batch/epoch balancing | `2` makes `effective_train_samples_per_epoch` about `2 * virtual_train_records`. |
| `preprocess.train_sampling.positive_repeat_factor` | `1` | int | `>= 1` | train | repeats | `4` expands positive and partial-positive virtual records 4x. |
| `preprocess.train_sampling.hard_negative_repeat_factor` | `1` | int | `>= 1` | train | repeats | `2` expands hard-negative virtual records 2x. |
| `preprocess.train_sampling.negative_repeat_factor` | `1` | int | `>= 1` | train | repeats | `1` leaves regular negative records unchanged. |
| `preprocess.train_sampling.positive_stride_factor` | `1.0` | float | `> 0`; effective stride is `max(1, int(stride * factor))` | train | window generation | With `stride=256`, `0.5` gives positive stride `128`; on 1024x1024 with 256 tiles this is 49 windows instead of 16. |
| `preprocess.train_sampling.hard_negative_stride_factor` | `1.0` | float | `> 0`; effective stride is `max(1, int(stride * factor))` | train | window generation | With `stride=256`, `0.5` scans hard-negative candidates every 128 px. |
| `preprocess.train_sampling.negative_stride_factor` | `1.0` | float | `> 0`; effective stride is `max(1, int(stride * factor))` | train | window generation | Values above/below `1.0` make regular negative candidate grid sparser/denser. |
| `preprocess.train_sampling.min_positive_pixels` | `1` | int | `>= 1` | train+val classification | tile classification | `10` makes masks with 1-9 foreground pixels `partial_positive` if partials are enabled. |
| `preprocess.train_sampling.include_partial_positive` | `true` | bool | `true` / `false` | train+val classification | tile classification | `false` makes sub-threshold non-empty tiles count as `negative`. |
| `preprocess.train_sampling.partial_positive_fraction` | `0.0` | float | `>= 0`; fraction of tile area | train+val classification | tile classification | `0.01` requires partial positives to cover at least 1% of tile pixels. |
| `preprocess.train_sampling.max_empty_tile_share` | `null` | float/null | `null` or `0.0..1.0` | train | record limiting | `0.35` caps `negative + hard_negative` share after base limiting and after repeats. |
| `preprocess.train_sampling.hard_negative_context_px` | `null` | int/null | `null` or `>= 0` | train | tile classification | `512` labels empty windows intersecting a positive bbox expanded by 512 px as `hard_negative`; `null` uses `tile_size // 2` and emits a warning. |
| `preprocess.train_sampling.batch_positive_fraction` | `null` | float/null | `null` or `0.0..1.0`; fractions are normalized if sum > 1 | train | batch/epoch balancing | `0.5` asks epoch indices to draw about 50% from positive plus partial-positive records. |
| `preprocess.train_sampling.batch_hard_negative_fraction` | `null` | float/null | `null` or `0.0..1.0`; fractions are normalized if sum > 1 | train | batch/epoch balancing | `0.25` asks epoch indices to draw about 25% from hard negatives. |
| `preprocess.train_sampling.batch_negative_fraction` | `null` | float/null | `null` or `0.0..1.0`; fractions are normalized if sum > 1 | train | batch/epoch balancing | `0.25` asks epoch indices to draw about 25% from regular negatives. |
| `preprocess.train_sampling.random_jitter.enabled` | `false` | bool | `true` / `false` | train | window generation | `true` shifts base train windows before repeats; validation never jitters. |
| `preprocess.train_sampling.random_jitter.max_shift_fraction` | `0.25` | float | `>= 0`; fraction of tile size | train | window generation | With `tile_size=768`, `0.25` allows up to 192 px shift in each direction, clipped to scene bounds. |
| `preprocess.train_sampling.random_jitter.keep_positive_min_pixels` | `1` | int | `>= 1` | train | tile classification | Positive/partial-positive jitter attempts fall back to the original window if foreground drops below this value. |

All parameters above are implemented in `mlsystem/src/data/virtual_tile_sampling.py`. The CLI/API preview reports `warnings` when a requested group is empty, fractions need normalization, or `hard_negative_context_px` falls back to `tile_size // 2`.

## Local Preview

Preview records without training:

```powershell
python scripts/debug_virtual_tile_dataset.py `
  --run-dir E:\Projects\MLSystemFork\MLSystem\mlsystem_runs\example `
  --images-dir E:\Projects\NSPD\Images\test `
  --config-yaml E:\Projects\MLSystemFork\MLSystem\debug_train_sampling.yaml `
  --output-json E:\Projects\MLSystemFork\MLSystem\outputs\virtual_tile_preview.json `
  --write-preview-overlays E:\Projects\MLSystemFork\MLSystem\outputs\debug_virtual_tile_sampling\overlays `
  --max-preview-overlays 20 `
  --max-scenes 3 `
  --max-records-preview 20
```

The script writes lightweight JSON and, when requested, small PNG overlays for visual debugging. It does not write image tiles, mask tiles, numpy arrays, or augmented files for training.

Run the synthetic self-check and parameter matrix:

```powershell
python scripts/debug_virtual_tile_sampling_selfcheck.py `
  --output-json outputs/debug_virtual_tile_sampling/parameter_matrix.json
```

## Local Image-Only HTML Report

When GeoTIFF files are already available locally but there is no `dataset_manifest.json` or annotation, build image-only reports:

```powershell
python scripts/debug_local_tile_report.py `
  --images-dir E:\Projects\NSPD\Images\test `
  --tile-size 768 `
  --stride 512 `
  --positive-stride-factor 0.5 `
  --hard-negative-stride-factor 0.5 `
  --negative-stride-factor 1.0 `
  --positive-repeat-factor 4 `
  --hard-negative-repeat-factor 2 `
  --negative-repeat-factor 1 `
  --max-empty-tile-share 0.35 `
  --max-overview-size 1600 `
  --max-tile-examples 24 `
  --max-augmentation-tiles 8 `
  --augmentation-mode all `
  --augmentation-seed 42
```

This creates `tile_sampling_index.html` in the images directory and one `tile_sampling_report.html` plus `scene_summary.json` per scene subdirectory. This mode is explicitly image-only: positive, partial-positive, hard-negative, and negative classes are not validated without annotation.

Augmentation report controls:

- `--augmentation-mode all`: render every supported operation.
- `--augmentation-mode individual`: render deterministic individual operations, excluding random train-like combinations.
- `--augmentation-mode production-groups`: render one or two representative operations for each production config group.
- `--augmentation-mode random-training`: render only train-like random combinations.
- `--augmentations flip_horizontal,rot90_90,gamma_low`: override the mode with an explicit comma-separated operation list.
- `--max-augmentation-tiles 8`: apply the full augmentation matrix only to the first selected tile examples, keeping the report size bounded.
- `--augmentation-seed 42`: seed deterministic debug previews and metadata.

Supported local report augmentation operations:

`original`, `flip_horizontal`, `flip_vertical`, `flip_horizontal_vertical`, `rot90_90`, `rot90_180`, `rot90_270`, `brightness`, `contrast`, `brightness_contrast`, `gamma_low`, `gamma_high`, `noise_low`, `noise_high`, `blur_light`, `blur_strong`, `cutout_small`, `cutout_medium`, `coarse_dropout`, `color_jitter`, `training_random_all_enabled_seed_1`, `training_random_all_enabled_seed_2`, `training_random_all_enabled_seed_3`.

The debug implementation mirrors production augmentation groups used by `real_train.py`: `flips`, `rot90`, `brightness_contrast`/`color_jitter`, `gamma`, `noise`, `blur`, `cutout`/`coarse_dropout`. Random train-like examples are visual preview equivalents and are marked with `matches_training_semantics=false` in JSON metadata when they do not reuse the exact torch batch code path. Geometric image+mask consistency is covered by synthetic unit tests; local GeoTIFF reports without annotation only validate RGB preview behavior.

## Annotated Tile Preparation Module

The reusable training tile preparation code lives in `mlsystem/src/tile_preparation/`. It is independent from Airflow, FastAPI, `real_train.py`, and pipeline stages. The public API accepts image and annotation paths, builds lightweight tile records, and reads raster windows plus rasterized masks lazily during iteration:

```python
from pathlib import Path
from mlsystem.src.tile_preparation import (
    AnnotationInput,
    SceneInput,
    TilePreparationConfig,
    build_tile_records,
    iter_training_tiles,
)

config = TilePreparationConfig(
    tile_size=768,
    stride=512,
    positive_stride_factor=0.5,
    hard_negative_stride_factor=0.5,
    negative_stride_factor=1.0,
    min_positive_pixels=1,
    max_empty_tile_share=0.35,
    positive_repeat_factor=4,
    hard_negative_repeat_factor=2,
    negative_repeat_factor=1,
    seed=42,
)

scene = SceneInput(image_path=Path("E:/Projects/NSPD/Images/test/scene.tif"), scene_id="scene")
annotation = AnnotationInput(
    geojson_path=Path("E:/Projects/NSPD/Images/test/deforestation.geojson"),
    annotation_crs="auto",
    allow_inferred_annotation_crs=True,
)

records = build_tile_records([scene], annotation, config)
for sample in iter_training_tiles([scene], annotation, config):
    image = sample.image
    mask = sample.mask
    record = sample.record
```

The iterator does not materialize all image/mask tiles. It only keeps lightweight records, then opens the raster window and rasterizes the mask for each yielded sample. Masks are `uint8` binary `{0,1}` before conversion to the requested output format.

## Annotated HTML Report

When one GeoTIFF and one GeoJSON are present in a local folder, build a full annotated report:

```powershell
python scripts/debug_annotated_tile_report.py `
  --input-dir E:\Projects\NSPD\Images\test `
  --tile-size 768 `
  --stride 512 `
  --positive-stride-factor 0.5 `
  --hard-negative-stride-factor 0.5 `
  --negative-stride-factor 1.0 `
  --positive-repeat-factor 4 `
  --hard-negative-repeat-factor 2 `
  --negative-repeat-factor 1 `
  --max-empty-tile-share 0.35 `
  --min-positive-pixels 1 `
  --hard-negative-context-px 384 `
  --annotation-crs auto `
  --allow-inferred-annotation-crs `
  --max-overview-size 1600 `
  --max-tile-examples 32 `
  --max-augmentation-tiles 8 `
  --augmentation-mode all `
  --augmentation-seed 42
```

The script writes `annotated_tile_sampling_index.html/json` in the input directory and `annotated_tile_sampling_report.html`, `annotated_scene_summary.json`, overview images, tile mask overlays, and augmentation mask overlays under the scene subdirectory. These are debug artifacts only and must not be committed.

CRS behavior:

- explicit `--annotation-crs` wins;
- GeoJSON `crs` is used when present;
- with `--allow-inferred-annotation-crs`, missing CRS can be inferred as EPSG:4326/EPSG:3857 or treated as raster CRS with a warning;
- if raster and annotation CRS differ, geometries are transformed to raster CRS before rasterization.

Cutout and coarse dropout match `real_train.py`: they modify image pixels only and keep the mask unchanged.

## Download A Few S3 Scenes

```powershell
python scripts/debug_download_s3_scenes.py `
  --manifest E:\Projects\MLSystemFork\MLSystem\mlsystem_runs\example\dataset_manifest.json `
  --images-uri s3://mlsystems/images/ `
  --output-dir E:\Projects\NSPD\Images\test `
  --max-scenes 3 `
  --prefer-positive true
```

On WSL/Linux the helper uses `/mnt/e/Projects/NSPD/Images/test` when that path exists. If S3 credentials are unavailable, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or configure the `mc` alias used by `MLSYSTEM_PIPELINE_CONFIG`.

## FastAPI Debug Endpoint

The endpoint is registered only when the env var is enabled:

```powershell
$env:MLSYSTEM_DEBUG_DATASET_ENDPOINTS="1"
python -m uvicorn mlsystem.src.api.app:app --host 127.0.0.1 --port 8088
```

Call it:

```powershell
$payload = @{
  dataset_manifest = "E:\Projects\MLSystemFork\MLSystem\mlsystem_runs\example\dataset_manifest.json"
  annotation = "E:\Projects\MLSystemFork\MLSystem\mlsystem_runs\example\dataset_annotation.geojson"
  images_dir = "E:\Projects\NSPD\Images\test"
  max_scenes = 3
  max_records_preview = 20
  config = @{
    preprocess = @{
      tile_size = 768
      stride = 512
      train_sampling = @{
        enabled = $true
        positive_stride_factor = 0.5
        hard_negative_stride_factor = 0.5
        negative_stride_factor = 1.0
        positive_repeat_factor = 4
        hard_negative_repeat_factor = 2
        negative_repeat_factor = 1
        max_empty_tile_share = 0.35
      }
    }
  }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8088/api/debug/virtual-dataset/preview `
  -ContentType "application/json" `
  -Body $payload
```

For local image-only preview:

```powershell
$payload = @{
  images_dir = "E:\Projects\NSPD\Images\test"
  tile_size = 768
  stride = 512
  stride_factors = @(1.0, 0.5, 0.25)
  max_scenes = 2
  max_records_preview = 20
  include_augmentation_catalog = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8088/api/debug/local-tile-report/preview `
  -ContentType "application/json" `
  -Body $payload
```

The local image-only endpoint does not return raster arrays or images. With `include_augmentation_catalog=true`, the response includes `augmentation_operations_supported`, `training_augmentation_keys_supported`, and `augmentation_report_available=false`; full HTML/PNG reports are generated only by the CLI script.

For annotated lightweight preview:

```powershell
$payload = @{
  input_dir = "E:\Projects\NSPD\Images\test"
  tile_size = 768
  stride = 512
  positive_stride_factor = 0.5
  hard_negative_stride_factor = 0.5
  negative_stride_factor = 1.0
  min_positive_pixels = 1
  max_scenes = 1
  max_records_preview = 20
  include_annotation_summary = $true
  include_augmentation_catalog = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8088/api/debug/annotated-tile-report/preview `
  -ContentType "application/json" `
  -Body $payload
```

The annotated endpoint is also gated by `MLSYSTEM_DEBUG_DATASET_ENDPOINTS=1`. It returns raster metadata, annotation metadata, tiling summary, classification summary, augmentation catalog, and warnings. It does not generate HTML and does not return raster or mask arrays.

## What To Check

In `summary`, check:

- `base_train_records`;
- `virtual_train_records`;
- `effective_train_samples_per_epoch`;
- `train_positive_tiles`, `train_partial_positive_tiles`, `train_hard_negative_tiles`, `train_negative_tiles`;
- `val_tile_count`, `val_positive_tiles`, `val_negative_tiles`;
- `positive_stride`, `hard_negative_stride`, `negative_stride`;
- `warnings`.

## Risks

Keep the split scene-level. Do not build validation from train-like virtual records. Do not enable random jitter or online augmentation for validation. For final runs, avoid train-to-val fallback and verify the MLflow params and `train_dataset_report.json` before comparing experiments.
