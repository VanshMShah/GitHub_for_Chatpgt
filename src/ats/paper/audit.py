from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only JSONL audit log for deterministic post-mortem replay."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, **payload: Any) -> None:
        record = {"event_type": event_type, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
