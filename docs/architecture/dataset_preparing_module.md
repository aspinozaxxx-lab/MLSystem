# dataset_preparing

## Назначение
`dataset_preparing` проверяет входные данные training pipeline: читает список сцен, сопоставляет сцены с изображениями, считает объекты по аннотациям, строит train/val split, формирует dataset manifest и вычисляет identity версии датасета.

## Public API
- `parse_scene_list_text(text)`: принимает текст `scenes.txt`, возвращает список строк сцен без комментариев и пустых строк.
- `build_scene_matching_report(entries, images, accept_threshold=0.92, ambiguous_margin=0.015, preferred_key_prefixes=None)`: принимает строки сцен и inventory изображений, возвращает отчет matching.
- `count_objects_per_scene(scene_names, scene_to_image, annotation_path, count_mode="auto", annotation_crs=None, allow_inferred_crs=True)`: считает объекты по сценам.
- `split_train_val_by_object_counts(counts, target_val_fraction=0.2, seed=42, target_val_objects=None, min_val_scenes=None, include_zero_object_scenes=True)`: строит object-balanced split.
- `split_manifest_scene_rows(items, train_fraction=0.75)`: строит legacy split по порядку manifest rows.
- `write_scene_object_counts_report(rows, output_path)`, `write_train_val_split_report(split, output_path)`, `write_split_outputs(rows, split, output_dir)`: пишут артефакты подготовки датасета.
- `compute_dataset_identity(dataset_manifest=None, inventory=None, scene_matching=None, class_name=None, class_slug=None, images_uri=None, layout_uri=None, scenes_uri=None, annotation_uri=None)`: вычисляет fingerprint/version/counts/scenes/git metadata для MLflow и training report.
- `dataset_identity_mlflow_payload(identity)`: возвращает `dataset.*` параметры для MLflow.

## Запрещенные пересечения
- Не создает tile datasets, dataloaders и augmentation: это `tile_preparation`.
- Не обучает модель: это `train`.
- Не управляет pipeline lifecycle и MLflow run: это `train_pipeline`.
- Не делает inference, vectorization, postprocess или pseudolabel export: это InferenceEngine/inference boundary.
- Не содержит FastAPI transport schemas или debug endpoints.
