import numpy as np
from flask import Flask, json

from iml.utils.io_utils import get_path

# path = get_path('frontend/dist/static', absolute=True)
# print("Static folder: {:s}".format(path))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)

app.config['FRONT_END_ROOT'] = get_path('front-end/build', absolute=True)
app.config['STATIC_FOLDER'] = get_path('front-end/build/static', absolute=True)
app.json_encoder = NumpyEncoder
