import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_APPLICATION_LOGGERS = (
    "document_processing",
    "graph_rag",
    "llm_providers",
    "wiki_base",
)


def configure_logging(
    level: str,
    *,
    log_directory: Path = Path("logs"),
    process_name: str = "application",
) -> Path:
    """Configure console and persistent application logs."""

    application_level = level.upper()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in _APPLICATION_LOGGERS:
        logging.getLogger(logger_name).setLevel(application_level)

    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = (log_directory / f"{process_name}.log").resolve()
    root_logger = logging.getLogger()
    existing_paths = {
        Path(handler.baseFilename).resolve()
        for handler in root_logger.handlers
        if isinstance(handler, RotatingFileHandler)
    }
    if log_path not in existing_paths:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(application_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s pid=%(process)d %(name)s "
                "%(filename)s:%(lineno)d %(funcName)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "persistent %s logs enabled at %s with level=%s",
        process_name,
        log_path,
        application_level,
    )
    return log_path
