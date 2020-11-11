import flask
from flask import Flask, request, render_template
import json
import main
import numpy as np
import pandas as pd


app = Flask(__name__)
cqa_val_lines = pd.read_csv('data/cosmos/valid.csv')
cqa_val_n = len(cqa_val_lines.index)
cqa_test_lines = open('data/cosmos/test.jsonl').readlines()
cqa_test_n = len(cqa_test_lines)

sqa_val_lines = open('data/socialiqa/dev.jsonl').readlines()
sqa_gld = open('data/socialiqa/dev-labels.lst').readlines()
sqa_val_n = len(sqa_val_lines)

mccarthy_prompt = open("data/mccarthy/prompt.txt").readline().strip()
mccarthy_lines = open("data/mccarthy/train.jsonl").readlines()
mccarthy_n = len(mccarthy_lines)

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
        if request.json["dataset"] == "cosmos":
            idx = np.random.randint(cqa_val_n)
            ln = cqa_val_lines.iloc[idx]
            dct = {"prompt": ln['context']}
            dct["choices"] = "\n".join([ln[f"answer{i}"] for i in range(4)])
            dct["question"] = ln['question']
            gld = f"answer{ln['label']}"
            dct["gold"] = f"Correct Answer: {ln[gld]}"
            return app.response_class(response=json.dumps(dct), status=200, mimetype='application/json')
        elif request.json["dataset"] == "social":
            idx = np.random.randint(sqa_val_n)
            ln = json.loads(sqa_val_lines[idx])
            ygld = int(sqa_gld[idx])-1
            dct = {"prompt": ln['context']}
            dct["choices"] = "\n".join([ln[f"answer{chr(ord('A')+i)}"] for i in range(3)])
            dct["question"] = ln['question']
            dct["gold"] = ln[f"answer{chr(ord('A')+ygld)}"]
            return app.response_class(response=json.dumps(dct), status=200, mimetype='application/json')
        else:
            idx = np.random.randint(mccarthy_n)
            ln = json.loads(mccarthy_lines[idx])
            ygld = int(ln["correct"])-1
            dct = {"prompt": mccarthy_prompt}
            dct["choices"] = "\n".join([ln[f"answer{chr(ord('A')+i)}"] for i in range(4)])
            dct["question"] = ln['question']
            dct["gold"] = ln[f"answer{chr(ord('A')+ygld)}"]            
            return app.response_class(response=json.dumps(dct), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print("ERROR HERE")
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

@app.route('/get_random_cqa_tst', methods=['post'])
def get_random_cqa_tst():
    try:
        print(request.json)
        idx = np.random.randint(cqa_test_n)
        ln = json.loads(cqa_test_lines[idx])
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
