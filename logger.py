import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'


def set_file_logger(filename):
    logging.root.handlers = []
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    # fh = logging.FileHandler(filename)
    # fh.setLevel(LOG_LEVEL)
    # fh.setFormatter(logging.Formatter(LOG_FORMAT))
    # logging.getLogger().addHandler(fh)


if len(logging.root.handlers) == 0:
    set_file_logger('poc-afz.log')


def get_logger(name=None):
    if name is None:
        return logging.getLogger()
    elif isinstance(name, str):
        return logging.getLogger(name)
    else:
        return logging.getLogger(name.__module__ + "." + name.__class__.__name__)
