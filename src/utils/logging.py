"""
Simple logger utility - prints & optionally writes to file
"""
import logging
import sys

def set_logger(name="lungnet", logfile=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
