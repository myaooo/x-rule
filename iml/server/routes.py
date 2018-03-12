from flask import request, abort, send_from_directory, safe_join, jsonify

from iml.server import app, get_model, available_models, HashableList
from iml.server.jsonify import model2json, model_data2json
from iml.server.helpers import model_metric, get_support, get_stream


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
    return send_from_directory(app.config['FRONT_END_ROOT'], 'index.html')


@app.route('/<string:model>', methods=['GET'])
def send_index(model):
    if model == 'service-worker.js':
        return send_from_directory(app.config['FRONT_END_ROOT'], 'service-worker.js')
    return send_from_directory(app.config['FRONT_END_ROOT'], 'index.html')


@app.route('/api/model', methods=['GET'])
def models():
    # model_name = request.args.get('name')
    # if model_name is None:
    return jsonify(available_models())
    # else:
    #     model_json = model2json(model_name)
    #     if model_json is None:
    #         abort(404)
    #     else:
    #         return model_json


@app.route('/api/model/<string:model_name>', methods=['GET'])
def model_info(model_name):
    # model_name = request.args.get('name')
    # if model_name is None:
    #     return jsonify(available_models())
    # else:
    model_json = model2json(model_name)
    if model_json is None:
        abort(404)
    else:
        return model_json


@app.route('/api/model_data/<string:model_name>', methods=['GET'])
def model_data(model_name):
    data_type = request.args.get('data', 'train')
    bins = int(request.args.get('bins', '20'))
    if model_name is None:
        abort(404)
    else:
        data_json = model_data2json(model_name, data_type, bins)
        if data_json is None:
            abort(404)
        else:
            return data_json


# @app.route('/api/data/<string:data_name>', methods=['GET'])
# def data(data_name):
#     data_type = request.args.get('data', 'train')
#     # is_train = not (request.args.get('isTrain') == 'false')
#     bins = request.args.get('bins')
#     # if bins is None:
#     #     bins = 15
#     if data_name is None:
#         abort(404)
#     else:
#         data_json = data2json(data_name, data_type, bins)
#         if data_json is None:
#             abort(404)
#         else:
#             return data_json


@app.route('/api/metric/<string:model_name>', methods=['GET'])
def metric(model_name):
    data = request.args.get('data', 'test')

    ret_json = model_metric(model_name, data)
    if ret_json is None:
        abort(404)
    else:
        return ret_json


@app.route('/api/support/<string:model_name>', methods=['GET'])
def support(model_name):
    data_type = request.args.get('data', 'train')
    support_type = request.args.get('support', 'simple')
    ret_json = get_support(model_name, data_type, support_type)
    if ret_json is None:
        abort(404)
    else:
        return ret_json


@app.route('/api/stream/<string:model_name>', methods=['GET'])
def stream(model_name):
    data_type = request.args.get('data', 'train')
    # per_class = request.args.get('class', 'true') == 'true'
    conditional = request.args.get('conditional', 'true') == 'true'
    bins = int(request.args.get('bins', '20'))
    ret_json = get_stream(model_name, data_type, conditional=conditional, bins=bins)
    if ret_json is None:
        abort(404)
    else:
        return ret_json


@app.route('/api/query/<string:model_name>', methods=['POST'])
def query(model_name):
    data_type = request.args.get('data', 'train')
    bins = int(request.args.get('bins', '20'))
    query_json = HashableList(request.get_json())
    if model_name is None:
        abort(404)
    else:
        data_json = model_data2json(model_name, data_type, bins, query_json)
        if data_json is None:
            abort(404)
        else:
            return data_json
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
        return get_model(name).predict(data)

"""
curl -i -X POST -H 'Content-Type: application/json' -d '[]' http://localhost:5000/api/query/rule-surrogate-breast_cancer-nn-20
"""