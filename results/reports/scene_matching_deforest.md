# Scene matching deforest

Дата проверки: 2026-04-26.

Источник:

- scenes: `s3://mlsystems/layouts/deforest/2026_04_25/scenes.txt`
- layout: `s3://mlsystems/layouts/deforest/2026_04_25/layout.geojson`
- images root: `s3://mlsystems/images/`

## Итог

| Metric | Value |
|---|---:|
| scenes.txt rows | 24 |
| available `.tif/.tiff` images | 61 |
| matched | 7 |
| missing | 17 |
| ambiguous | 0 |
| missing caused by matcher | 0 |
| really absent from S3 | 17 |

Matcher был расширен: теперь он пишет `scene_matching_report.json` с normalized names, signature по `KANOPUS + date + time + SCN`, двумя лучшими кандидатами и причиной решения.

## Matched

Все 7 matched scenes имеют `normalized_exact` score `1.0`:

| scenes.txt entry | S3 object |
|---|---|
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07` | `images/incoming/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07.tif` |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN04` | `images/incoming/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN04.tif` |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03` | `images/incoming/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03.tif` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN06` | `images/incoming/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN06.tif` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN05` | `images/incoming/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN05.tif` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN04` | `images/incoming/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN04.tif` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN03` | `images/incoming/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN03.tif` |

## Missing

Все 17 missing имеют низкие best-candidate scores `0.680851..0.757895`; ближайшие S3-кандидаты имеют другие даты/время/сцены. Это реальные отсутствующие снимки, а не ошибка matcher.

Missing entries:

- `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN03`
- `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN02`
- `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN01`
- `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN03`
- `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN02`
- `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN01`
- `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN03`
- `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN02`
- `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN01`
- `KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN02`
- `KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN01`
- `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN03`
- `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN02`
- `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN01`
- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN03`
- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN02`
- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN01`

## Вывод

Чтобы получить 24/24 matched scenes, нужно дозагрузить эти 17 исходных TIFF в `s3://mlsystems/images/incoming/<delivery>/`.
