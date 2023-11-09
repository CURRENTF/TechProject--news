import time
from loguru import logger as log

from config import config


class Logger(object):
    def __init__(self):
        self.logger = log

        self.logger.add(f"{config.LOG_PATH}/tech_project_log_{time.strftime('%Y-%m-%d')}.log",
                        rotation="1 week",
                        encoding="utf-8",
                        enqueue=True,
                        retention="30 days",
                        compression="zip")

    def trace(self, *args, **kwargs):
        return self.logger.trace(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return self.logger.critical(*args, **kwargs)

    def process_logging(self, message):
        self.info(message)

    def output_logging(self, message):
        if isinstance(message, dict):
            for k, v in message.items():
                self.info(f"{k}: {v}")
        else:
            self.info(message)

    def error_logging(self, message):
        self.error(message)

    def warning_logging(self, message):
        self.warning(message)


logger = Logger()
