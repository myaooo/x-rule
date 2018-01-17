"""
ModelManager
"""

import os
import pickle

from iml.utils.logging_utils import get_logger
from iml.models import FILE_EXTENSION, ModelBase

logger = get_logger(__name__)


class ModelManager:

    def __init__(self):
        self.models = {}

    def load(self, path):
        for e in os.walk(path):
            files = e[-1]
            for file in files:
                self.load_model(file)

    def load_model(self, filename):
        model = pickle.load(filename)
        if isinstance(model, ModelBase):
            self.models[model.name] = model

        else:
            logger.warning(f"The file {filename} is not a supported .{FILE_EXTENSION} model!")
