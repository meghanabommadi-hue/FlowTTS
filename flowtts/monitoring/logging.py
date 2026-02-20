"""Structured logging setup for FlowTTS.

This module centralizes structlog configuration so that the gateway,
worker, and decoder all emit logs in a consistent format.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import structlog


def _build_processors(json_logs: bool) -> Iterable[Any]:
    """Return the list of structlog processors used across the app."""
    base: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_logs:
        base.append(structlog.processors.JSONRenderer())
    else:
        base.append(structlog.dev.ConsoleRenderer())
    return base


def configure_logging(*, json_logs: bool = False) -> None:
    """Configure structlog for FlowTTS.

    Call this once during application startup (e.g. in ``main.py``) before
    importing modules that call ``structlog.get_logger()``.
    """
    structlog.configure(
        processors=_build_processors(json_logs),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(),  # type: ignore[arg-type]
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a structlog logger, optionally bound to a specific name."""
    return structlog.get_logger(name) if name is not None else structlog.get_logger()

