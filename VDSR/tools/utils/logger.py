import logging
import time
from datetime import timedelta
import sys
import os

initialized_logger = {}

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        # round() 返回浮点数x的四舍五入值
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            elapsed_seconds
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath=None, is_console=False, change_stdout_stderr=False):
    """
    Create a logger.
    """
    if filepath==None and not is_console:
        raise Exception('Logger has no handler')
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_foder = os.path.dirname(filepath)
        if not os.path.exists(file_foder):
            # exist_ok：是否在目录存在时触发异常。
            # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
            # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
            os.makedirs(file_foder)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    if is_console:
        # create console handler and set level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    if is_console:
        logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    if change_stdout_stderr:
        sys.stdout = LoggerWriter(logger.debug)
        sys.stderr = LoggerWriter(logger.warning)

    return logger


def _init_logger(logger, log_level=logging.DEBUG, filepath=None, is_console=False, change_stdout_stderr=False):
    """
    Create a logger.
    """
    if filepath==None and not is_console:
        raise Exception('Logger has no handler')
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_folder = os.path.dirname(filepath)
        if file_folder == '':
            file_folder = './'
        if not os.path.exists(file_folder):
            # exist_ok：是否在目录存在时触发异常。
            # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
            # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
            os.makedirs(file_folder)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    if is_console:
        # create console handler and set level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger.handlers = []
    logger.setLevel(log_level)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    if is_console:
        logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    if change_stdout_stderr:
        sys.stdout = LoggerWriter(logger.debug)
        sys.stderr = LoggerWriter(logger.warning)


def get_logger(logger_name='base', log_level=logging.DEBUG, log_file=None, is_console=False, change_stdout_stderr=False):
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    else:
        _init_logger(logger, log_level, log_file, is_console, change_stdout_stderr)
        initialized_logger[logger_name] = True
        return logger

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)