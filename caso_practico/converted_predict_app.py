import pickle

from flask import Flask, jsonify, request


def predict_single(customer, dv, model):
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[:, 1]
    return (y_pred[0] >= 0.5, y_pred[0])


app = Flask('converted_predict')

with open('../models/converted-model.pck', 'rb') as f:
    dv, model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    converted, prediction = predict_single(customer, dv, model)

    result = {
        'converted': bool(converted),
        'converted_probability': float(prediction),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
