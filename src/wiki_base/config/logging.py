import logging

_APPLICATION_LOGGERS = (
    "document_processing",
    "graph_rag",
    "llm_providers",
    "wiki_base",
)


def configure_logging(level: str) -> None:
    """Configure application logs without enabling noisy dependency traces."""

    application_level = level.upper()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in _APPLICATION_LOGGERS:
        logging.getLogger(logger_name).setLevel(application_level)
