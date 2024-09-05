import json
import logging


def read_json_config(file_path):
    """Read and return the JSON configuration from the given file path."""
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


def configure_logging():
    """Configure the logging module to log messages to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(processName)s] [%(name)s] - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger
