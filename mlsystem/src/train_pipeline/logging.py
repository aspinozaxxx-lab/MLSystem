from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from ..api.security import mask_text


class TeeTextIO:
    def __init__(self, original: TextIO, files: list[TextIO], buffer: list[str]) -> None:
        self.original = original
        self.files = files
        self.buffer = buffer

    def write(self, text: str) -> int:
        masked = mask_text(text) or ""
        self.buffer.append(masked)
        self.original.write(masked)
        self.original.flush()
        for fp in self.files:
            fp.write(masked)
            fp.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        for fp in self.files:
            fp.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tee_stage_logs(run_dir: Path, stage: str) -> Iterator[None]:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = (log_dir / "pipeline.log").open("a", encoding="utf-8")
    stage_log = (log_dir / f"{stage}.log").open("a", encoding="utf-8")
    stdout_buffer: list[str] = []
    stderr_buffer: list[str] = []
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = TeeTextIO(old_stdout, [pipeline_log, stage_log], stdout_buffer)  # type: ignore[assignment]
    sys.stderr = TeeTextIO(old_stderr, [pipeline_log, stage_log], stderr_buffer)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        pipeline_log.close()
        stage_log.close()
        _write_tail(log_dir / "stdout_tail.txt", "".join(stdout_buffer))
        _write_tail(log_dir / "stderr_tail.txt", "".join(stderr_buffer))


def _write_tail(path: Path, text: str, max_chars: int = 20000) -> None:
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    path.write_text((existing + text)[-max_chars:], encoding="utf-8")
