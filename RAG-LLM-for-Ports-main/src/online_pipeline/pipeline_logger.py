# src/online_pipeline/pipeline_logger.py
"""
Centralized logging configuration for the RAG pipeline.

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)

Call `setup_pipeline_logging()` once at application startup
(e.g., in demo scripts or evaluation runner) to configure output.

Log levels:
- DEBUG:   detailed variable dumps (retrieved docs, SQL rows, full paths)
- INFO:    key decisions and flow milestones (router decision, sources selected)
- WARNING: fallback triggered, missing evidence, API degradation
- ERROR:   execution failures, LLM errors, DB errors
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False

# Consistent format: timestamp | level | module | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_pipeline_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
) -> None:
    """
    Configure logging for the entire online_pipeline package.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file path to write logs to
        console: Whether to also print to stderr (default True)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("online_pipeline")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_pipeline_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper: returns a logger under the online_pipeline namespace.
    Ensures all pipeline modules share the same handler configuration.
    """
    return logging.getLogger(f"online_pipeline.{name}")
