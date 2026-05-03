# Changelog

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
- Provereny lokalnye komandy `compileall` i `unittest`.
