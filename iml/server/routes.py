from iml.server import app
from flask import request


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/model', methods=['GET'])
def model():
    name = request.args.get('name')
    if name is None:
        return
