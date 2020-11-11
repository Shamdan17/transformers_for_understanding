import flask
from flask import Flask, request, render_template
import json
import main
import numpy as np
import pandas as pd


app = Flask(__name__)
val_lines = pd.read_csv('valid.csv')
val_n = len(val_lines.index)
test_lines = open('test.jsonl').readlines()
test_n = len(test_lines)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help.html', methods=['get'])
def help():
    return render_template('help.html')


@app.route('/get_end_predictions', methods=['post'])
def get_prediction_eos():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        # input_text += ' <mask>'
        q = request.json['question']
        c = request.json['choices']
        top_k = request.json['top_k']
        res = main.get_all_predictions(input_text,q, c, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print("ERROR HERE")
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


@app.route('/get_random_cqa_val', methods=['post'])
def get_random_cqa_val():
    try:
        idx = np.random.randint(val_n)
        ln = val_lines.iloc[idx]
        dct = {"prompt": ln['context']}
        dct["choices"] = "\n".join([ln[f"answer{i}"] for i in range(4)])
        dct["question"] = ln['question']
        gld = f"answer{ln['label']}"
        dct["gold"] = f"Correct Answer: {ln[gld]}"
        return app.response_class(response=json.dumps(dct), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print("ERROR HERE")
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

@app.route('/get_random_cqa_tst', methods=['post'])
def get_random_cqa_tst():
    try:
        idx = np.random.randint(test_n)
        ln = json.loads(test_lines[idx])
        dct = {"prompt": ln['context']}
        dct["choices"] = "\n".join([ln[f"answer{i}"] for i in range(4)])
        dct["question"] = ln['question']
        dct["gold"] = ""
        return app.response_class(response=json.dumps(dct), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print("ERROR HERE")
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


@app.route('/get_mask_predictions', methods=['post'])
def get_prediction_mask():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        top_k = request.json['top_k']
        res = main.get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8383, use_reloader=False)
