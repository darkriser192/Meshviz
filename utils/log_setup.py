import logging

def setup_logger(name="meshviz", level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger