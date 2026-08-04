import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from wiki_base.config.logging import configure_logging


def test_configure_logging_writes_persistent_debug_log(tmp_path: Path) -> None:
    """Persistent logging records detailed application messages once."""

    log_path = configure_logging(
        "DEBUG",
        log_directory=tmp_path,
        process_name="test-process",
    )
    configure_logging(
        "DEBUG",
        log_directory=tmp_path,
        process_name="test-process",
    )

    root_logger = logging.getLogger()
    handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename).resolve() == log_path
    ]
    assert len(handlers) == 1

    logging.getLogger("wiki_base.test").debug("retrieval trace marker")
    handlers[0].flush()
    assert "retrieval trace marker" in log_path.read_text(encoding="utf-8")

    root_logger.removeHandler(handlers[0])
    handlers[0].close()
