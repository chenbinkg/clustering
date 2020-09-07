import logging
import os


class ESDAKemLogging(object):
    """
    This is a singleton served as the logging facility of
    probe hierarchical clustering.
    """

    _instance = None  # shared instance

    def __new__(cls, *args, **kwargs):
        # make sure the instance object only allocated once
        if cls._instance is None:
            cls._instance = super(ESDAKemLogging, cls).__new__(cls)
            return cls._instance
        else:
            return None

    def __init__(self, logging_file_name="", output_dir=".", logging_level=logging.INFO):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if logging_file_name == "":
            # get a default logger which print informarion to screen
            self.logger = logging.getLogger()
            ch = logging.StreamHandler()
            ch.setLevel(logging_level)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging_level)
        else:
            self.logger = logging.getLogger()
            fh = logging.FileHandler(os.path.join(output_dir, logging_file_name + ".log"))
            fh.setLevel(logging_level)
            self.logger.addHandler(fh)
            self.logger.setLevel(logging_level)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    @classmethod
    def getInstance(cls):
        if cls._instance is not None:
            return cls._instance
        else:
            raise Exception("ESDAKemLogging has not yet been initialized.")
