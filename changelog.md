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
- Dlya MLflow experiments dobavleny experiment tags `class_name` i `mlsystem.class_name`, chtoby klass byl viden na urovne Experiments.
- Provereny lokalnye komandy `compileall` i `unittest`.
