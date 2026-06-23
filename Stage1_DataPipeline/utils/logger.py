"""
logger.py — Dual-Sink Logging Configuration
============================================

Provides a factory function that creates a Python ``logging.Logger``
with two handlers:

1.  **Console (StreamHandler)** — coloured, human-readable output at
    INFO level so the operator can monitor progress in real time.
2.  **Rolling File (RotatingFileHandler)** — DEBUG-level records
    written to ``logs/pipeline_step1.log``, rotating at 5 MB with
    3 backup copies so we never lose evidence of a crash.

Usage::

    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Reading CASME II Excel …")

Design Decision:
    We avoid third-party logging libraries (loguru, structlog) to
    satisfy the thesis constraint of minimising external dependencies
    and keeping the audit trail readable by the committee.

Author  : Addhyan
Stage   : 1 — Data Pipeline
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_filename: str = "pipeline_step1.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB per file
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with console + rotating-file
    sinks.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    log_dir : Path, optional
        Directory for the log file.  Created if it does not exist.
        Defaults to ``Stage1_DataPipeline/logs/``.
    log_filename : str
        Base name of the log file.
    console_level : int
        Minimum severity printed to stdout.
    file_level : int
        Minimum severity written to the log file.
    max_bytes : int
        Maximum size (bytes) of a single log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """

    logger = logging.getLogger(name)

    # ── Guard: if handlers are already attached, return immediately
    #    (prevents duplicate output when get_logger is called twice
    #    for the same name in notebooks / interactive sessions).
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter.

    # Make stdout resilient on Windows consoles that default to cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # ── Formatter common to both sinks ──────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 1. Console handler ──────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── 2. Rotating file handler ────────────────────────────────────
    if log_dir is None:
        # Default: <Stage1_DataPipeline>/logs/
        log_dir = Path(__file__).resolve().parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.debug(
        "Logger '%s' initialised.  Console=%s  File=%s  → %s",
        name,
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
        log_path,
    )

    return logger
