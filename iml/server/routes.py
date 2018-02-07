from flask import request, abort, send_from_directory, safe_join, jsonify

from iml.server import app, get_model, available_models
from iml.server.jsonify import model2json, data2json


@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory(safe_join(app.config['STATIC_FOLDER'], 'js'), path)


@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory(safe_join(app.config['STATIC_FOLDER'], 'css'), path)


@app.route('/static/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory(safe_join(app.config['STATIC_FOLDER'], 'fonts'), path)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/api/model/<string:model_name>', methods=['GET'])
def model_info(model_name):
    # model_name = request.args.get('name')
    if model_name is None:
        return jsonify(available_models())
    else:
        model_dict = model2json(model_name)
        if model_dict is None:
            abort(404)
        else:
            return jsonify(model_dict)


@app.route('/api/data/<string:data_name>', methods=['GET'])
def data(data_name):
    is_train = not (request.args.get('isTrain') == 'false')
    bins = request.args.get('bins')
    # if bins is None:
    #     bins = 15
    if data_name is None:
        abort(404)
    else:
        data_dict = data2json(data_name, is_train, bins)
        if data_dict is None:
            abort(404)
        else:
            return jsonify(data_dict)


# @app.route('/api/models/<string:model_name>', methods=['GET'])
# def model_info(model_name):
#     model = get_model(model_name)
#     return jsonify(model2json(model))


@app.route('/api/predict', methods=['POST'])
def predict():
    name = request.args.get('name')
    data = request.args.get('data')
    if name is None:
        abort(404)
    else:
        get_model(name).predict()