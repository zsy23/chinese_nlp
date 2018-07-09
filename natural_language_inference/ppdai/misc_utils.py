import logging

logger = logging.getLogger('text_similarity')


def log(msg):
    global logger
    logger.info(msg)