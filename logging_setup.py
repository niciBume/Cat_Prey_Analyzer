import logging
from logging.handlers import RotatingFileHandler
import os, gzip, shutil

def setup_logging(log_filename, max_log_size, backup_count, log_level_str="INFO"):
    """
    Configure logging for the current process (main or subprocess):
      - File rotation with gzip compression
      - Level and formatting
      - Removes all previous handlers (avoids duplicate logs)
    """
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    log_handler = RotatingFileHandler(
        log_filename, maxBytes=max_log_size, backupCount=backup_count
    )

    # Compress old log files after rotation
    class GzipRotator:
        def __call__(self, source, dest):
            with open(source, 'rb') as f_in, gzip.open(dest + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(source)
    log_handler.rotator = GzipRotator()
    log_handler.namer = lambda name: name

    formatter = logging.Formatter('%(asctime)s [%(levelname)s][PID %(process)d]: %(message)s',
                                  datefmt='%x-%X')
    log_handler.setFormatter(formatter)

    log_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")
    logger.setLevel(log_level)
    logger.addHandler(log_handler)
