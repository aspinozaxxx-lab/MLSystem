# InferenceEngine overlap inventory

InferenceEngine содержит inference-specific реализации, которые концептуально похожи на части MLSystem, но принадлежат inference runtime:

- tiling/windows для inference;
- probability map и stitching;
- vectorization/postprocessing;
- pseudolabel export и compatibility artifacts.

В этой задаче InferenceEngine production-code не рефакторится. `inference_pipeline` использует InferenceEngine как внешний исполнитель через API/client и не копирует эти реализации в MLSystem modules.
