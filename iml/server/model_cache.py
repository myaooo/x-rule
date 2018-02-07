import os
from logging import getLogger

from iml import Config
from iml.models import ModelInterface, load_model, FILE_EXTENSION
from iml.utils.io_utils import get_ext, file_exists, get_path, json2dict


logger = getLogger(__name__)


def model_name2file(model_name):
    return Config.get_path(Config.model_dir(), model_name + FILE_EXTENSION)


class ModelCache:
    """A wrapper that """

    def __init__(self):
        # self.name2file = {}
        self.cache = {}
        self.model2dataset = {}

    def init(self, config_path=None):
        if config_path is None:
            config_path = Config.config_dir()
        config = json2dict(config_path)
        if 'models' in config:
            for model_config in config['models']:
                self.model2dataset[model_config['model']] = model_config['dataset']
        # print(path)
        return self

    def load_model(self, model_name):
        filename = model_name2file(model_name)
        model = load_model(filename)
        if isinstance(model, ModelInterface):
            self.cache[model_name] = model
            return model
        else:
            raise RuntimeError("Mal-format! Cannot load model file {}!".format(filename))

    def get_model(self, model_name):
        if model_name in self.cache:
            return self.cache[model_name]
        return self.load_model(model_name)


_cache = ModelCache().init()


def get_model(model_name):
    return _cache.get_model(model_name)


def available_models():
    return list(_cache.model2dataset.keys())


def get_model_data(model_name):
    return _cache.model2dataset[model_name]


def register_model_dataset(model_name, dataset):
    _cache.model2dataset[model_name] = dataset
