"""Central logging setup: clean application logs, quiet third-party noise."""

from __future__ import annotations

import logging
import os
import warnings

_CONFIGURED = False


def quiet_ultralytics() -> None:
    """Force the Ultralytics logger quiet. It resets its own level to INFO on
    import, so this is called again right after that import."""
    logging.getLogger("ultralytics").setLevel(logging.ERROR)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure one consistent log format and silence noisy libraries.

    Ultralytics logs a 'half is deprecated' warning on every prediction, and
    Werkzeug logs every HTTP request (including UI polling); both are muted so
    the console shows only meaningful SkySentinel events.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    os.environ.setdefault("YOLO_VERBOSE", "False")   # ask Ultralytics to stay quiet
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy, lvl in (
        ("werkzeug", logging.WARNING),
        ("ultralytics", logging.ERROR),
        ("urllib3", logging.WARNING),
        ("PIL", logging.WARNING),
        ("matplotlib", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(lvl)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
