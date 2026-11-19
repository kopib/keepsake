"""Lightweight, color-aware logging configuration for keepsake."""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

_RESET = "\033[0m"
_DIM = "\033[2m"
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[2;37m",
    logging.INFO: "\033[36m",
    logging.WARNING: "\033[33;1m",
    logging.ERROR: "\033[31;1m",
    logging.CRITICAL: "\033[41;37;1m",
}

# Attach NullHandler by default so library imports stay silent until configured.
logging.getLogger("keepsake").addHandler(logging.NullHandler())


class _ColorFormatter(logging.Formatter):
    """Concise `HH:MM:SS | LEVEL | message` formatter with optional ANSI colors."""

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:<7}"
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if not self.use_color:
            return f"{timestamp} | {level} | {message}"

        color = _LEVEL_COLORS.get(record.levelno, "")
        return f"{_DIM}{timestamp}{_RESET} | {color}{level}{_RESET} | {message}"


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def get_logger(name: str = "keepsake") -> logging.Logger:
    """Return a logger scoped under the `keepsake` namespace."""
    if name == "keepsake" or name.startswith("keepsake."):
        return logging.getLogger(name)
    return logging.getLogger(f"keepsake.{name}")


def configure_logging(
    *,
    level: int = logging.INFO,
    verbose: bool = False,
    quiet: bool = False,
    stream: TextIO | None = None,
    force_color: bool | None = None,
) -> logging.Logger:
    """Configure the root `keepsake` logger once for CLI or script runs."""
    target_stream = stream if stream is not None else sys.stderr
    resolved_level = logging.DEBUG if verbose else (logging.WARNING if quiet else level)
    use_color = _supports_color(target_stream) if force_color is None else force_color

    root = logging.getLogger("keepsake")
    root.setLevel(resolved_level)
    root.propagate = False

    # Replace previous keepsake stream handlers so repeated calls never duplicate lines.
    for handler in list(root.handlers):
        if not isinstance(handler, logging.NullHandler):
            root.removeHandler(handler)

    handler = logging.StreamHandler(target_stream)
    handler.setLevel(resolved_level)
    handler.setFormatter(_ColorFormatter(use_color=use_color))
    root.addHandler(handler)
    return root
