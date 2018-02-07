"""
Configurations of the module
"""

import os
import json

# from iml.models import FILE_EXTENSION
# from iml.utils.io_utils import get_ext

# Constants


class Config:

    ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../'))
    MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models'))
    CONFIG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'config.json'))
    ENV = 'production'

    @classmethod
    def config_dir(cls, config_dir=None):
        if config_dir is not None:
            cls.CONFIG_DIR = config_dir
            return cls
        return cls.CONFIG_DIR

    @classmethod
    def root_dir(cls, root_dir=None):
        if root_dir is not None:
            cls.ROOT_DIR = root_dir
            return cls
        return cls.ROOT_DIR

    @classmethod
    def model_dir(cls, model_dir=None):
        if model_dir is not None:
            cls.MODEL_DIR = model_dir
            return cls
        return cls.MODEL_DIR

    @classmethod
    def get_path(cls, path, filename=None, absolute=False):
        """
        A helper function that get the real/abs path of a file on disk, with the project dir as the base dir.
        Note: there is no checking on the illegality of the args!
        :param path: a relative path to ROOT_DIR, optional file_name to use
        :param filename: an optional file name under the path
        :param absolute: return the absolute path
        :return: return the path relative to the project root dir, default to return relative path to the called place.
        """
        _p = os.path.join(cls.ROOT_DIR, path)
        if filename:
            _p = os.path.join(_p, filename)
        if absolute:
            return os.path.abspath(_p)
        return os.path.relpath(_p)

    # @classmethod
    # def load(cls, model_dir=None):
    #     if model_dir is None:
    #         model_dir = cls.model_dir()
    #     for file in os.listdir(model_dir):
    #         if os.path.isfile(file) and get_ext(file) == FILE_EXTENSION:
    #             cls.models |= {file}

    @classmethod
    def mode(cls, new_mode=None):
        if new_mode is None:
            return cls.ENV
        elif new_mode in ['production', 'development']:
            cls.ENV = new_mode
            return cls
        else:
            raise ValueError("Unknown mode {:s}".format(new_mode))


def init_config(config_file='config.json'):
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        if 'root_dir' in config:
            Config.root_dir(
                Config.get_path(config['root_dir'], absolute=True))
        if 'model_dir' in config:
            Config.model_dir(
                Config.get_path(config['model_dir'], absolute=True))
