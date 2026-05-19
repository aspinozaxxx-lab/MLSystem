from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..metrics.api import PixelMetricAccumulator
from ._checkpoints import load_initial_checkpoint, save_checkpoint
from ._device import resolve_device
from ._history import write_history
from ._losses import segmentation_loss_components
from ._models import build_model
from ._optimizers import build_optimizer
from ._progress import EarlyStopping, emit_progress
from ._schedulers import build_scheduler
from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent, TrainProgressSink, TrainRequest, TrainResult


class _WeightedLossAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weight = 0

    def update(self, values: dict[str, Any], weight: int) -> None:
        weight = int(weight)
        self.weight += weight
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * weight

    def averages(self, prefix: str) -> dict[str, float]:
        if self.weight <= 0:
            return {f"{prefix}/{key}": 0.0 for key in sorted(self.sums)}
        return {f"{prefix}/{key}": float(value / self.weight) for key, value in self.sums.items()}


def run_training(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult:
    cfg = request.config
    if request.train_dataloader is None:
        raise TrainError("TrainRequest.train_dataloader is required.")
    if request.val_dataloader is None:
        raise TrainError("TrainRequest.val_dataloader is required.")

    device = resolve_device(cfg.device, require_gpu=cfg.require_gpu)
    model = build_model(cfg.model_name, cfg.input_channels, cfg.output_channels, cfg.base_channels).to(device)
    if cfg.initial_checkpoint_path:
        load_initial_checkpoint(model, cfg.initial_checkpoint_path, device=device, strict=cfg.initial_checkpoint_strict)

    optimizer = build_optimizer(
        model,
        {
            "optimizer": cfg.optimizer,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
    )
    scheduler = build_scheduler(optimizer, {"scheduler": cfg.scheduler}, cfg.epochs)
    stopper = EarlyStopping(cfg.early_stopping_patience) if cfg.early_stopping_patience else None

    output_dir = Path(request.output_dir) if request.output_dir is not None else None
    history_rows: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_f1 = float("-inf")
    best_iou = float("-inf")
    early_stopped = False
    started = time.time()

    emit_progress(progress_sink, TrainProgressEvent(stage="started", message="training started"))
    for epoch in range(1, max(1, int(cfg.epochs)) + 1):
        epoch_started = time.time()
        train_metrics = _run_train_epoch(
            model,
            request.train_dataloader,
            optimizer,
            device,
            loss_config=cfg.loss,
            max_batches=cfg.max_train_batches,
        )
        val_metrics = _run_val_epoch(
            model,
            request.val_dataloader,
            device,
            loss_config=cfg.loss,
            threshold=cfg.metric_threshold,
            thresholds=cfg.metric_thresholds,
            max_batches=cfg.max_val_batches,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_duration_sec": round(time.time() - epoch_started, 4),
        }
        history_rows.append(row)
        current_f1 = float(row.get("val/pixel_f1", 0.0))
        current_iou = float(row.get("val/iou", row.get("val/pixel_iou", 0.0)))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_iou = current_iou
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        emit_progress(progress_sink, TrainProgressEvent(stage="epoch_completed", epoch=epoch, metrics=_numeric_metrics(row)))
        if scheduler is not None:
            scheduler.step()
        if stopper is not None and stopper.update(current_f1, epoch):
            early_stopped = True
            break

    artifacts: list[str] = []
    checkpoint: CheckpointArtifact | None = None
    if output_dir is not None:
        history_paths = write_history(output_dir, history_rows)
        artifacts.extend(str(path) for path in history_paths)
        checkpoint_path = output_dir / f"{cfg.model_name}.pt"
        if best_state is not None:
            model.load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path,
            model,
            {
                "model_name": cfg.model_name,
                "best_epoch": best_epoch,
                "best_metric": "val/pixel_f1",
                "best_metric_value": best_f1 if best_f1 != float("-inf") else 0.0,
                "epochs_completed": len(history_rows),
            },
        )
        checkpoint = CheckpointArtifact(
            path=str(checkpoint_path),
            model_name=cfg.model_name,
            epoch=best_epoch,
            metric_name="val/pixel_f1",
            metric_value=best_f1 if best_f1 != float("-inf") else 0.0,
        )
        artifacts.append(str(checkpoint_path))

    total_duration_sec = round(time.time() - started, 4)
    last = history_rows[-1] if history_rows else {}
    best_row = max(history_rows, key=lambda row: float(row.get("val/pixel_f1", 0.0)), default={})
    training_summary = _training_summary(
        history_rows=history_rows,
        best_row=best_row,
        last_row=last,
        best_f1=best_f1 if best_f1 != float("-inf") else 0.0,
        total_duration_sec=total_duration_sec,
        early_stopped=early_stopped,
    )
    if output_dir is not None:
        diagnostics_path = output_dir / "training_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(
                _training_diagnostics(
                    history_rows=history_rows,
                    best_row=best_row,
                    last_row=last,
                    best_f1=best_f1 if best_f1 != float("-inf") else 0.0,
                    total_duration_sec=total_duration_sec,
                    early_stopped=early_stopped,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(str(diagnostics_path))
        threshold_sweep = _threshold_sweep_summary(history_rows)
        if threshold_sweep:
            threshold_sweep_path = output_dir / "threshold_sweep_summary.json"
            threshold_sweep_path.write_text(
                json.dumps(threshold_sweep, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            artifacts.append(str(threshold_sweep_path))
    emit_progress(progress_sink, TrainProgressEvent(stage="completed", metrics=training_summary))
    return TrainResult(
        status="done",
        model_name=cfg.model_name,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        epochs_completed=len(history_rows),
        best_epoch=best_epoch,
        best_val_pixel_f1=best_f1 if best_f1 != float("-inf") else 0.0,
        best_val_iou=best_iou if best_iou != float("-inf") else 0.0,
        last_val_pixel_f1=float(last.get("val/pixel_f1", 0.0)),
        checkpoint=checkpoint,
        history=[EpochMetrics(epoch=int(row["epoch"]), metrics=_numeric_metrics(row)) for row in history_rows],
        training_time_sec=total_duration_sec,
    )


def result_to_dict(result: TrainResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["checkpoint_path"] = result.checkpoint.path if result.checkpoint else None
    payload["best_val_iou"] = result.best_val_iou
    payload["last_epoch_metrics"] = result.history[-1].metrics if result.history else {}
    return payload


def _training_summary(
    *,
    history_rows: list[dict[str, Any]],
    best_row: dict[str, Any],
    last_row: dict[str, Any],
    best_f1: float,
    total_duration_sec: float,
    early_stopped: bool,
) -> dict[str, float | int]:
    epochs_total = len(history_rows)
    epoch_time_sec = (
        sum(float(row.get("epoch_duration_sec", 0.0)) for row in history_rows) / epochs_total
        if epochs_total
        else 0.0
    )
    payload: dict[str, float | int] = {
        "best_val_pixel_f1": best_f1,
        "epochs_completed": epochs_total,
        "average_epoch_time_sec": round(epoch_time_sec, 4),
        "training_time_sec": total_duration_sec,
    }
    early = _float_or_none(1 if early_stopped else 0)
    best_epoch = _float_or_none(best_row.get("epoch"))
    last_f1 = _float_or_none(last_row.get("val/pixel_f1"))
    if early is not None:
        payload["early_stopped"] = early
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if last_f1 is not None:
        payload["last_val_pixel_f1"] = last_f1
    return {key: value for key, value in payload.items() if value is not None}


def _threshold_sweep_summary(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds: dict[str, list[dict[str, float | int]]] = {}
    best_by_epoch: list[dict[str, float | int]] = []
    for row in history_rows:
        epoch = int(row.get("epoch", 0) or 0)
        for key, value in row.items():
            marker = "_at_threshold_"
            if marker not in key:
                continue
            metric, suffix = key.split(marker, 1)
            if not metric.endswith("pixel_f1"):
                continue
            number = _float_or_none(value)
            if number is None:
                continue
            thresholds.setdefault(suffix, []).append({"epoch": epoch, "pixel_f1": number})
        best_threshold = _float_or_none(row.get("val/best_threshold"))
        best_f1 = _float_or_none(row.get("val/pixel_f1_best_threshold"))
        if best_threshold is not None and best_f1 is not None:
            best_by_epoch.append({"epoch": epoch, "threshold": best_threshold, "pixel_f1": best_f1})
    if not thresholds and not best_by_epoch:
        return {}
    return {"thresholds": thresholds, "best_by_epoch": best_by_epoch}


def _training_diagnostics(
    *,
    history_rows: list[dict[str, Any]],
    best_row: dict[str, Any],
    last_row: dict[str, Any],
    best_f1: float,
    total_duration_sec: float,
    early_stopped: bool,
) -> dict[str, float | int]:
    train_loss = _float_or_none(last_row.get("train/loss_total"))
    val_loss = _float_or_none(best_row.get("val/loss_total"))
    train_f1 = _float_or_none(last_row.get("train/pixel_f1"))
    payload: dict[str, float | int | None] = {
        "val_pixel_precision": _float_or_none(best_row.get("val/pixel_precision") or best_row.get("val/precision")),
        "val_pixel_recall": _float_or_none(best_row.get("val/pixel_recall") or best_row.get("val/recall")),
        "val_pixel_iou": _float_or_none(best_row.get("val/pixel_iou") or best_row.get("val/iou")),
        "val_pixel_accuracy": _float_or_none(best_row.get("val/pixel_accuracy") or best_row.get("val/accuracy")),
        "val_loss_total": val_loss,
        "train_loss_total": train_loss,
        "train_val_loss_gap": (train_loss - val_loss) if train_loss is not None and val_loss is not None else None,
        "train_pixel_f1": train_f1,
        "train_val_f1_gap": (train_f1 - best_f1) if train_f1 is not None else None,
        "learning_rate": _float_or_none(last_row.get("learning_rate")),
        "best_epoch": _float_or_none(best_row.get("epoch")),
        "early_stopped": 1 if early_stopped else 0,
        "train_batches": _float_or_none(last_row.get("train/batches")),
        "val_batches": _float_or_none(best_row.get("val/batches")),
        "train_samples": _float_or_none(last_row.get("train/samples")),
        "val_samples": _float_or_none(best_row.get("val/samples")),
        "samples_per_sec": (
            (_float_or_none(last_row.get("train/samples")) or 0.0) / total_duration_sec
            if total_duration_sec > 0
            else None
        ),
        "batches_per_sec": (
            (_float_or_none(last_row.get("train/batches")) or 0.0) / total_duration_sec
            if total_duration_sec > 0
            else None
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _run_train_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    losses = _WeightedLossAccumulator()
    batches = 0
    samples = 0
    for batch in loader:
        if max_batches is not None and batches >= int(max_batches):
            break
        x_cpu, y_cpu = _batch_xy(batch)
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        components = segmentation_loss_components(logits, y, loss_config)
        components["loss_total"].backward()
        optimizer.step()
        batch_size = int(x.shape[0])
        losses.update({key: value.detach().item() for key, value in components.items()}, weight=batch_size)
        batches += 1
        samples += batch_size
    return {**losses.averages("train"), "train/batches": float(batches), "train/samples": float(samples)}


@torch.no_grad()
def _run_val_epoch(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    loss_config: dict[str, Any],
    threshold: float,
    thresholds: list[float] | None,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    losses = _WeightedLossAccumulator()
    metrics = PixelMetricAccumulator(threshold=threshold)
    sweep_thresholds = _normalized_thresholds(thresholds, configured_threshold=threshold)
    sweep_metrics = {item: PixelMetricAccumulator(threshold=item) for item in sweep_thresholds}
    batches = 0
    samples = 0
    for batch in loader:
        if max_batches is not None and batches >= int(max_batches):
            break
        x_cpu, y_cpu = _batch_xy(batch)
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True).float()
        logits = model(x)
        components = segmentation_loss_components(logits, y, loss_config)
        batch_size = int(x.shape[0])
        losses.update({key: value.detach().item() for key, value in components.items()}, weight=batch_size)
        metrics.update_from_logits(logits, y)
        for accumulator in sweep_metrics.values():
            accumulator.update_from_logits(logits, y)
        batches += 1
        samples += batch_size
    metric_payload = metrics.metrics()
    sweep_payload: dict[str, float | int] = {}
    best_sweep_threshold: float | None = None
    best_sweep_metrics: dict[str, float | int] | None = None
    for sweep_threshold, accumulator in sweep_metrics.items():
        current = accumulator.metrics()
        suffix = _threshold_suffix(sweep_threshold)
        for key, value in current.items():
            if key == "threshold":
                continue
            sweep_payload[f"val/{key}_at_threshold_{suffix}"] = value
        if best_sweep_metrics is None or float(current.get("pixel_f1", 0.0)) > float(best_sweep_metrics.get("pixel_f1", 0.0)):
            best_sweep_threshold = sweep_threshold
            best_sweep_metrics = current
    if best_sweep_threshold is not None and best_sweep_metrics is not None:
        sweep_payload.update(
            {
                "val/best_threshold": best_sweep_threshold,
                "val/pixel_f1_best_threshold": float(best_sweep_metrics.get("pixel_f1", 0.0)),
                "val/precision_best_threshold": float(best_sweep_metrics.get("pixel_precision", best_sweep_metrics.get("precision", 0.0))),
                "val/recall_best_threshold": float(best_sweep_metrics.get("pixel_recall", best_sweep_metrics.get("recall", 0.0))),
                "val/pixel_iou_best_threshold": float(best_sweep_metrics.get("pixel_iou", best_sweep_metrics.get("iou", 0.0))),
            }
        )
    return {
        **losses.averages("val"),
        **{f"val/{key}": value for key, value in metric_payload.items()},
        **sweep_payload,
        "val/batches": float(batches),
        "val/samples": float(samples),
    }


def _normalized_thresholds(thresholds: list[float] | None, *, configured_threshold: float) -> list[float]:
    if thresholds is None:
        return []
    normalized: list[float] = []
    for item in thresholds:
        try:
            value = max(0.0, min(1.0, float(item)))
        except (TypeError, ValueError):
            continue
        if value not in normalized:
            normalized.append(value)
    configured = max(0.0, min(1.0, float(configured_threshold)))
    if configured not in normalized:
        normalized.append(configured)
    return normalized


def _threshold_suffix(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _batch_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[1], batch[2]
        if len(batch) >= 2:
            return batch[0], batch[1]
    raise TrainError("Training batches must be (x, y), (indices, x, y), or {'x': x, 'y': y}.")


def _numeric_metrics(row: dict[str, Any]) -> dict[str, float | int]:
    return {key: value for key, value in row.items() if isinstance(value, (int, float))}


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None
