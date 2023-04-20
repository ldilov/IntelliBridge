import logging
import sys
import warnings

import transformers
from loguru import logger

from kernel.persistence.storage.file_manager import FileManager

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("safetensors").setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

transformers.logging.disable_default_handler()

logger_format = (
    "<green>{time:HH:mm:ss}</green> | "
    "{level.icon} | "
    "<level>{level.name}</level> | "
    "<cyan>{module}</cyan> | "
    "<level>{message}</level>"
)

logger_format_context = (
    "<green>{time:HH:mm:ss}</green> | "
    "{level.icon} | "
    "<level>{level.name}</level> | "
    "<cyan>{module}</cyan> | "
    "<level>{message}: <magenta>{extra[context]}</magenta></level>"
)

logger.remove()

logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    diagnose=True,
    backtrace=True,
    enqueue=True,
    format=logger_format,
    filter=lambda record: record["extra"].get("context", None) is None
)

logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    diagnose=True,
    backtrace=True,
    enqueue=True,
    format=logger_format_context,
    filter=lambda record: record["extra"].get("context", False)
)


def format_record_level(record, width):
    record["level"].name = record["level"].name.center(width)
    return record


def format_record_module(record, width):
    return record.update({"module": f"{record['thread'].name}::{record['module']}".ljust(width)})


# Text formatting
logger = logger.patch(lambda record: format_record_module(record, 24))
logger = logger.patch(lambda record: format_record_level(record, 7))

disabled_logging = [
    "safetensors"
]

for lib in disabled_logging:
    logger.disable(lib)

__all__ = ['logger']
