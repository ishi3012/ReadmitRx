import logging
from pathlib import Path


def configure_logging(
    log_name: str = "readmitrx", log_file: str = "run.log"
) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        filename=log_dir / log_file,
        filemode="w",
    )

    return logging.getLogger(log_name)
