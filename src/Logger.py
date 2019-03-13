import logging
import os


def get_logger(f_path):
    level = logging.INFO
    format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    handlers = [logging.FileHandler(os.path.join(f_path, 'model.log')), logging.StreamHandler()]
    logging.basicConfig(level=level, datefmt='%Y-%m-%d %H:%M:%S', format=format, handlers=handlers)
    logger = logging.getLogger('Deep-BGT')
    return logger

