"""
Logging configuration for the Clinical QA System.

Configures loguru with:
  - Console output (coloured, human-readable)
  - Rotating file output
  - MLflow metric logging hook for key events
  - Structured JSON output option for production

Call setup_logging() once at application startup.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings


def setup_logging(
    level: str = "",
    log_file: Optional[str] = None,
    json_logs: bool = False,
    enable_mlflow_sink: bool = False,
) -> None:
    """
    Configure loguru for the application.

    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR). Defaults to settings.log_level.
        log_file: Path to log file. Defaults to settings.log_file.
        json_logs: Emit JSON-formatted logs (for production/ELK).
        enable_mlflow_sink: Also forward WARNING+ logs to MLflow as tags.
    """
    level = level or settings.log_level
    log_path = Path(log_file or settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # ── Console handler ───────────────────────────────────────────────────────
    if json_logs:
        fmt = (
            '{{"time":"{time:YYYY-MM-DD HH:mm:ss}", '
            '"level":"{level}", '
            '"module":"{module}", '
            '"message":"{message}"}}'
        )
    else:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=not json_logs,
        enqueue=True,
    )

    # ── File handler (rotating 10 MB, keep 7 days) ────────────────────────────
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} — {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
    )

    # ── MLflow sink (optional) ────────────────────────────────────────────────
    if enable_mlflow_sink:
        try:
            import mlflow

            def mlflow_sink(message):
                record = message.record
                if record["level"].no >= 30:  # WARNING and above
                    try:
                        mlflow.set_tag(
                            f"log_{record['level'].name.lower()}",
                            record["message"][:250],
                        )
                    except Exception:
                        pass

            logger.add(mlflow_sink, level="WARNING", format="{message}")
        except ImportError:
            pass

    logger.debug(f"Logging configured: level={level}, file={log_path}")


def get_query_logger(query_id: str):
    """Return a logger instance bound with a specific query ID for tracing."""
    return logger.bind(query_id=query_id)


# Apply default setup when module is imported directly
def _auto_setup() -> None:
    """Light setup with no file handler — used when settings aren't fully loaded yet."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
