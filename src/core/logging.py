"""
Apex Intelligence Engine V5 — Structured Logging
==================================================
JSON-structured logging with separate streams for different concerns.
Replaces the old loguru console-only output with production-grade logging.

Usage:
    from src.core.logging import get_logger

    logger = get_logger("apex.models.training")
    logger.info("training_started", symbol="BTCUSDT", features=23, samples=50000)
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

from src.core.config import ROOT_DIR

# ── Log Directory ────────────────────────────────────────────────────────────
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _configure_structlog() -> None:
    """Configure structlog for JSON-structured output."""

    # Shared processors for all loggers
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Development: human-readable colored output
    # Production: JSON output for ELK/Datadog ingestion
    if sys.stderr.isatty():
        # Development mode — pretty console output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )
    else:
        # Production mode — JSON lines
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING and above (suppresses DEBUG/INFO noise)
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Run configuration on import
_configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger bound to a specific module name.

    Args:
        name: Logger name, typically "apex.module.submodule"

    Returns:
        A bound structlog logger instance.

    Examples:
        logger = get_logger("apex.data.binance")
        logger.info("candle_received", symbol="BTCUSDT", close=68000.0)
        logger.error("connection_failed", reason="timeout", retry_in=5)
    """
    return structlog.get_logger(name)
