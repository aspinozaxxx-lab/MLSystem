# Changelog

## 2026-05-15

- Optimizirovan `tile_preparation`: dobavlen facade DataLoader path s workers/prefetch/pin_memory/persistent_workers, STRtree geometry index, `uint8_255` normalization fast path i profiling data-wait/tile-prep metrik.
- `real_train.py` pereveden na `TilePreparationFacade.train_dataloader` / `val_dataloader`; CUDA transfer ispolzuet `non_blocking=True`, validation ostaetsya bez augmentation/repeats.
- Dobavlen `scripts/benchmark_tile_preparation_dataloader.py` i tests dlya DataLoader workers, geometry index, normalization i real_train integration.
- Utochneny DataLoader wait metrics: `data/batch_wait_sec` ostalsya alias na total, dobavleny total/mean/median/p95/max, samples/sec i batches/sec; pipeline worker stderr/stdout teper pishutsya v run logs, a `WorkerExited` poluchaet diagnostic payload.
- `real_train.py` otklyuchaet effective `persistent_workers` dlya pere-sozdavaemyh po epoch DataLoader, chtoby ne ostavlyat stale worker processes; requested/effective znacheniya logiruyutsya otdelno.
- `mlsystem-api` container poluchil `shm_size=${MLSYSTEM_API_SHM_SIZE:-1gb}`, chtoby DataLoader workers mogli peredavat 512px batch tensor bez `/dev/shm` OOM.
- Dobavlen lokalnyy `scripts/profile_tile_training_local.py`: korotkiy profiler batch_wait, tile-prep function timings, CPU->GPU transfer, forward/loss/backward/optimizer step s JSON/CSV/Markdown output.
- `tile_preparation` pereveden na `SceneFootprint`: records stroyatsya tolko po fakticheskomu footprint snimka, boundary valid mask rasterizuetsya iz footprint, per-window `read_valid_mask` v build records ubrany; dobavlen `scripts/benchmark_tile_record_build.py`.
- `tile_preparation` poluchil internal `SceneAdjacencyIndex` i `TileMosaicPlan`: mosaic vyzyvaetsya tolko dlya boundary records s geometriqueski poleznymi sosedyami, `WarpedVRT` sozdaetsya lazy, neighbor valid mask beretsya iz footprint.
- `real_train.py` poluchil opt-in profiling `profile_step_timing` / `MLSYSTEM_TRAIN_STEP_PROFILE` i sborka worker `tile_prep_profile` metadata pri `MLSYSTEM_TILE_PREP_PROFILE=1`, chtoby v history/MLflow videt CPU->GPU, forward, backward, optimizer i tile-prep breakdown pri DataLoader workers.
- DataLoader workers/prefetch/pin/persistent policy sdelany internal defaults `tile_preparation`: Linux default workers=16, Windows fallback=0, trace keys ignoriruyutsya kak deprecated; env override ostalsya tolko dlya diagnostiki.
- Zafiksirovany architecture docs dlya module boundaries, `tile_preparation`, `mlflow_adapter`, `pipeline_runner`, tile/preprocessing inventory i pipeline/orchestration inventory; dobavlen runbook zapuska experimentov bez Airflow.

## 2026-05-07

- Ispravlen deploy `pipeline.server.yaml`: service deploy teper vosstanavlivaet `mlsystem/configs/pipeline.server.yaml` posle rsync s `--delete`, a compose yavno peredaet `MLSYSTEM_PIPELINE_CONFIG`.
- Dodelana proverka razmetok dlya TXT so strokami-papkami: raskrytie ostaetsya v backend/data layer, a frontend tolko pokazyvaet gotovyy pipeline report.
- Uproshchena stranitsa `annotation-check`: v forme ostalis tolko GeoJSON/JSON i TXT, technical warnings perevedeny v ponyatnye soobshcheniya, zero-object scenes pokazany kak info, stseny sortiruyutsya po object_count.
- Razdelen frontend deploy na bystryy `frontend-site` i otdelnyy `frontend-ansible` dlya nastroyek.

## 2026-05-01

- Vosstanovleny baseline-dannye dlya `IRKUTSK-pseudolabel-unet-r50-t512-v4` i `GPU-deforest-segformer-b0-t1024-gpu-v13`.
- Dobavleny lokalnye baseline-artefakty v `artifacts/baselines`.
- V train loop vklyucheny realnye augmentatsii iz `train.augmentations`: flips, rot90, brightness_contrast/color_jitter, gamma, noise, blur, cutout/coarse_dropout.
- Dobavlena podderzhka `pseudolabel.images_uri`, chtoby trenirovat model na deforest dataset, a itogovuyu psevdo-razmetku strogo stroitt po `images/kanopus/irkutsk/`.
- Ispravleno logirovanie MLflow: `*.accepted.geojson` teper logiruetsya v artifacts nezavisimo ot lokalnogo limita razmera artifact.
- Dobavlena podderzhka segformer_b2/segformer_b3, kratkie epoch-metriki epoch/pixel_f1, epoch/pixel_iou, epoch/sec dlya grafikov MLflow.
- Uskorena psevdo-razmetka: parallelizovana vektorizatsiya scen, dobavleno osvobozhdenie GPU posle inference i zashchita ot slishkom shumnyh raw polygon kandidatov.
- Dobavleny Airflow resource pools `gpu_training`, `cpu_heavy`, `cpu_light`, `io_light`; DAG stages poluchili pool i pool_slots.
- Dobavlen resource-monitoring dlya Airflow stages: CPU load, RAM i GPU snapshot pishutsya v stage resources JSON i task logs.
- Optimizirovana postobrabotka psevdo-razmetki: dobavlen ranniy area-prefilter pri vectorization i bystryy poryadok candidate search dlya shumnyh fallback scen.
- Psevdo-razmetka razdelena na Airflow stages: GPU inference pishet per-scene probability maps, CPU vectorize/postprocess chitaet ih iz run-dir i bolshe ne derzhit GPU slot.
- Per-scene probability maps sohranyayutsya bez compression, chtoby GPU stage ne blokiroval GPU slot na dolgoy CPU-kompressii.
- Inference OOM fallback teper lovit ne tolko torch.cuda.OutOfMemoryError, no i torch.AcceleratorError/RuntimeError s CUDA memory allocation, chtoby avtomaticheski umenshat batch.
- Dlya bolshih SegFormer 1024 inference default batch i gpu_forward_concurrency sdelany konservativnee, chtoby b2/b3 ne padali po CUBLAS/CUDA allocation.
- Dlya MLflow experiments dobavleny experiment tags `class_name` i `mlsystem.class_name`, chtoby klass byl viden na urovne Experiments.
- Provereny lokalnye komandy `compileall` i `unittest`.

## 2026-05-03

- Dobavlen realny Triton ONNX export dlya segmentatsionnyh modeley i batch inference client dlya Triton HTTP.
- Psevdo-razmetka all-images dlya deforest perevedena na Triton backend s `deforest_segformer_b1_t1024`.
- Ispravlena utechka RAM v Airflow inference: bounded scene futures bolshe ne derzhat probability maps zavershennyh scen.
- Dobavlen skip failed scenes v pseudolabel inference, chtoby odin pustoy/bitiy TIFF ne valil ves DAG; failed scenes pishutsya v coverage report.
- S3 cache dlya all-images inference teper ochishchaetsya posle kazhdoy sceny, a probability intermediates sohranyayutsya kak compact uint8.
- Dobavlen env-flag `MLSYSTEM_DISABLE_S3_FILE_CACHE` dlya otdelnyh streaming scenariev, no osnovnoy all-images put ispolzuet rabochiy boto3 cache s purge.
- Triton pereveden v explicit model-control mode: tyazhelaya model zagruzhaetsya pered inference stage i vygruzhaetsya posle nego, chtoby osvobozhdat VRAM mezhdu rabotami.
- Ochishcheny bolshie Airflow intermediates `pseudolabel_scene_results` i `vectorization_work` na GPU servere; finalnye geojson/summaries/checkpoints ne udalalis.
- V `finalize_mlflow_run` dobavlena avtomaticheskaya ochistka runtime intermediates posle logirovaniya finalnyh artifacts; ee mozhno otklyuchit cherez `pseudolabel.cleanup_intermediates=false`.
- Dobavlen Airflow maintenance DAG `mlsystem_maintenance_cleanup`: ezhednevno chistit starye runtime intermediates, lokalnyy S3 cache i starye Airflow logs bez udaleniya MinIO/MLflow/finalnyh artifacts.
- Provereny lokalnye komandy `compileall` i `unittest`.

## 2026-05-15 JupyterLab

- Dobavleny frontend card `JupyterLab`, env `FRONTEND_JUPYTER_URL` i nginx route `/jupyter/` s frontend `auth_request`, WebSocket proxy i upstream `mlsystem-gpu-jupyter:8888`.
- Otkluchena vnutrennyaya token/password-avtorizatsiya JupyterLab na servere: dostup ostalsya tolko cherez frontend-auth `/jupyter/`, XSRF proverka Jupyter ne otklyuchalas.
