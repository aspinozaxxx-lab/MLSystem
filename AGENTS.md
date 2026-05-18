Правила закрепленные в docs\architecture\architecture.md и связанных документах обязательны к исполнению!

# Experiment Tuning Policy

For model tuning, Codex must not create or use tuning supervisor scripts.

Forbidden:
- `tune_*.py`
- `*_tuning*.py`
- `*_forever*.py`
- `*_supervisor*.py`
- adaptive tuning loops
- random-search loops
- background tuning processes
- shell loops
- cron/systemd/tmux/screen/nohup based tuning automation
- any script that repeatedly launches Airflow experiments

Required workflow:
1. Launch exactly one experiment manually through Airflow CLI/API.
2. Wait for the Airflow DAG run to finish.
3. Verify that the result appears in MLflow.
4. Read MLflow metrics and Airflow status artifacts.
5. Analyze `best_val_pixel_f1`, `last_val_pixel_f1`, precision, recall, checkpoint, overfit/stability.
6. Decide the next experiment manually.
7. Launch the next single Airflow experiment manually.
8. Repeat as Codex operator, not through a script.

Every experiment must:
- use Airflow;
- log to MLflow;
- optimize pixel-level F1;
- have pseudo-labeling disabled;
- have a human-readable hypothesis;
- be recorded in the manual experiment journal.
