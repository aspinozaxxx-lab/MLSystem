# storage

## Назначение
`storage` предоставляет единый public API для local JSON/text helpers и S3/MinIO доступа.

## Public API
- `read_json(path, default=None)`: читает JSON-файл или возвращает `default`.
- `write_json(path, payload)`: атомарно пишет JSON.
- `append_text(path, text)`: добавляет текст в файл.
- `s3_parts(uri)`: разбирает `s3://bucket/key`.
- `s3_client(config)`: создает S3 client по runtime config.
- `list_s3_objects(config, uri, suffixes=None)`: перечисляет local/S3 объекты.
- `read_s3_text(config, uri)`, `read_s3_json(config, uri)`: читает local/S3 payload.
- `find_layout_files(config, layout_uri, scenes_file, annotation_file)`: находит scenes/annotation в layout.
- `raster_path_for_s3_key(config, key)`: возвращает путь к cached raster или `/vsis3`.
- `build_s3_layout_status(config)`: строит status S3 layout.
- `check_s3(config, write_test=False)`: проверяет S3 endpoint/credentials.

## Запрещенные пересечения
- Не парсит dataset semantics и scene matching: это `dataset_preparing`.
- Не владеет pipeline lifecycle.
- Не логирует в MLflow.
- Внешние production-модули используют `storage.api`, а не внутренние `storage.s3`/`storage.local_io`.
