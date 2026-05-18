# Модуль inference_pipeline

## Назначение
- единственный оркестратор для inference. Он предназначен только для запуска создания псевдоразметки. 

Он отвечает за:

- создание run;
- хранение trace;
- run status;
- stage lifecycle;
- logs;
- progress;
- cancellation;
- final summary;
- MLflow metadata logging через `mlflow_adapter`.
