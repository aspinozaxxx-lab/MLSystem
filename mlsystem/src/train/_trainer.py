from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..metrics.api import PixelMetricAccumulator, WeightedLossAccumulator
from ._checkpoints import load_initial_checkpoint, save_checkpoint
from ._device import resolve_device
from ._history import write_history
from ._losses import segmentation_loss_components
from ._models import build_model
from ._optimizers import build_optimizer
from ._progress import EarlyStopping, emit_progress
from ._schedulers import build_scheduler
from .contracts import CheckpointArtifact, EpochMetrics, TrainError, TrainProgressEvent, TrainProgressSink, TrainRequest, TrainResult


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

    last = history_rows[-1] if history_rows else {}
    mlflow_params = {
        "train.model_name": cfg.model_name,
        "train.epochs": cfg.epochs,
        "train.optimizer": cfg.optimizer,
        "train.learning_rate": cfg.learning_rate,
        "train.weight_decay": cfg.weight_decay,
        "train.metric_threshold": cfg.metric_threshold,
    }
    mlflow_metrics = {
        "train/epochs_completed": len(history_rows),
        "train/duration_sec": round(time.time() - started, 4),
        "val/best_pixel_f1": best_f1 if best_f1 != float("-inf") else 0.0,
        "val/best_iou": best_iou if best_iou != float("-inf") else 0.0,
        "val/last_pixel_f1": float(last.get("val/pixel_f1", 0.0)),
    }
    emit_progress(progress_sink, TrainProgressEvent(stage="completed", metrics=mlflow_metrics))
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
        mlflow_params=mlflow_params,
        mlflow_metrics=mlflow_metrics,
        mlflow_artifacts=artifacts,
    )


def result_to_dict(result: TrainResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["checkpoint_path"] = result.checkpoint.path if result.checkpoint else None
    payload["best_val_iou"] = result.best_val_iou
    payload["last_epoch_metrics"] = result.history[-1].metrics if result.history else {}
    return payload


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
    losses = WeightedLossAccumulator()
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
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    losses = WeightedLossAccumulator()
    metrics = PixelMetricAccumulator(threshold=threshold)
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
        batches += 1
        samples += batch_size
    metric_payload = metrics.metrics()
    return {
        **losses.averages("val"),
        **{f"val/{key}": value for key, value in metric_payload.items()},
        "val/batches": float(batches),
        "val/samples": float(samples),
    }


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
