# Ручной тест split_train_val_test

Этот handtest проверяет локальную подготовку train/val выборок:

- очистку списка сцен от отсутствующих TIFF/TIF;
- подсчет объектов разметки по сценам;
- детерминированный object-balanced train/val split;
- запись человекочитаемых отчетов.

Production-логика находится в `mlsystem/src/dataset_preparing/_dataset_split.py`. Handtest только вызывает этот код и не содержит копии алгоритма.

## Что лежит во входе

- `input/deforestation.txt` - очищенный список сцен.
- `input/*.geojson` - ровно один GeoJSON с разметкой.
- TIFF/TIF снимки в репозиторий не копируются.

Исходный общий список и GeoJSON берутся из:

```bat
tests\handtests\input
```

## Папка снимков

По умолчанию скрипты рекурсивно ищут снимки в:

```bat
E:\Projects\NSPD\Images\Kanopus
```

Путь можно переопределить переменной окружения:

```bat
set IMAGES_DIR=D:\path\to\Kanopus
tests\handtests\split_train_val_test\prepare_input.bat
tests\handtests\split_train_val_test\run.bat
```

## Как запускать

Из корня репозитория:

```bat
tests\handtests\split_train_val_test\prepare_input.bat
tests\handtests\split_train_val_test\run.bat
```

`prepare_input.bat`:

- берет `tests\handtests\input\deforestation.txt`;
- сверяет список с `IMAGES_DIR`;
- создает backup исходного списка;
- перезаписывает очищенный общий список;
- пишет `tests\handtests\input\deforestation.missing_removed.txt`;
- копирует очищенный txt и единственный GeoJSON в `split_train_val_test\input`;
- не копирует снимки.

`run.bat`:

- очищает и пересоздает `output`;
- проверяет, что в `input` есть ровно один GeoJSON;
- запускает CLI `python -m mlsystem.src.dataset_preparing._dataset_split split`;
- пишет отчеты.

## Что смотреть в output

- `output/scene_object_counts.txt` - `scene_name`, `object_count`, `image_path`, `matched_by`.
- `output/train_val_split.txt` - списки train/val и summary.
- `output/train.txt` - список train scenes.
- `output/val.txt` - список val scenes.
- `output/split_summary.json` - машинночитаемая сводка split: количество файлов и объектов, доли val, число сцен без объектов, `count_modes`, `crs_source`, warnings.

## Важные ограничения

- Снимки TIFF/TIF не должны попадать в repo.
- Если в `input` нет GeoJSON или их больше одного, `run.bat` должен остановиться с понятной ошибкой.
- CRS policy: сначала используется явно заданный CRS production config, затем CRS из GeoJSON. Если GeoJSON не содержит CRS и подсчет объектов переходит в geometry fallback, CRS может быть определен эвристически, это попадает в warnings и `split_summary.json` как `crs_source=inferred:*`. Для production лучше задавать CRS явно.
