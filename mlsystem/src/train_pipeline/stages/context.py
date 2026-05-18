from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StageContext:
    stage_id: str
    run_id: str
    config: Any
    raw_conf: dict[str, Any]
    status_dir: Path
    store: Any
    logger: logging.Logger
