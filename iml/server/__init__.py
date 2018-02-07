from flask import Flask
from iml.server.model_cache import get_model, available_models, get_model_data

app = Flask(__name__)

from iml.server.routes import *
