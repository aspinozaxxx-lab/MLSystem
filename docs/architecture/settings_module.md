# settings

## Назначение
`settings` загружает runtime config MLSystem и описывает публичные config DTO.

## Public API
- `load_config(path=None)`: читает YAML config. Параметр `path` задает явный путь; если он не передан, используется `MLSYSTEM_PIPELINE_CONFIG` или default config.
- `ensure_storage_layout(config)`: создает runtime storage directories для переданного `PipelineConfig`.
- `PipelineConfig`: DTO runtime config.
- `MLflowConfig`, `StorageConfig`, `S3PathsConfig`, `WebConfig`: вложенные DTO.

## Запрещенные пересечения
- Не запускает pipeline и не создает runs.
- Не читает и не пишет MLflow.
- Не содержит training/dataset/inference бизнес-логику.
- Не является compatibility wrapper для старых root config файлов.
