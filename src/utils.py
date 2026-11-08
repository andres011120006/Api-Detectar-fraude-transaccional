"""Structured JSON logging and small helpers."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

Status = Literal["APPROVED", "FLAGGED", "REJECTED"]

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON (CloudWatch / Cloud Logging friendly)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit JSON to stdout (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True


def risk_to_status(score: float, risk_threshold: float, flag_threshold: float) -> Status:
    """Map a fraud probability to a business decision."""
    if score >= risk_threshold:
        return "REJECTED"
    if score >= flag_threshold:
        return "FLAGGED"
    return "APPROVED"
