from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StageCheck:
    name: str
    status: str
    message: str


@dataclass
class StageReport:
    stage_id: str
    status: str = "success"
    checks: list[StageCheck] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload

    def to_stage_payload(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "summary": self.summary or self._default_summary(),
            "checks": [asdict(check) for check in self.checks],
            "counters": self.counters,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
            "details": self.details,
            "stage_report": self.to_dict(),
        }
        if self.errors:
            payload["errors"] = self.errors
            payload["error"] = "; ".join(self.errors)
        return payload

    def to_airflow_log(self, *, max_items: int = 1000) -> str:
        header_status = "FAILED" if self.status == "failed" else "summary"
        lines = [f"=== {self.stage_id} {header_status} ==="]
        for check in self.checks:
            lines.append(f"{check.name} - {check.status.upper()}: {check.message}")
        if self.counters:
            lines.append("counters:")
            for key, value in self.counters.items():
                lines.append(f"- {key}: {value}")
        if self.warnings:
            lines.append(f"warnings: {len(self.warnings)}")
            for warning in self.warnings[:max_items]:
                lines.append(f"- {warning}")
            if len(self.warnings) > max_items:
                lines.append(f"- ... {len(self.warnings) - max_items} more warnings")
        else:
            lines.append("warnings: 0")
        if self.errors:
            lines.append(f"errors: {len(self.errors)}")
            for error in self.errors[:max_items]:
                lines.append(f"- {error}")
        else:
            lines.append("errors: 0")
        if self.artifacts:
            lines.append("artifacts:")
            for name, path in self.artifacts.items():
                lines.append(f"- {name}: {path}")
        return "\n".join(lines)

    def _default_summary(self) -> str:
        if self.status == "failed":
            return f"{self.stage_id} failed"
        return f"{self.stage_id} completed"


class StageFailure(RuntimeError):
    def __init__(self, message: str, report: StageReport) -> None:
        super().__init__(message)
        self.report = report
