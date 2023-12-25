import pickle
from flask import Flask, request, jsonify
import tensorflow as tf
from model import KNN

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route('/nearest_predict', methods=['POST'])
def nearest_predict():
    data = request.get_json()
    lat = data["lat"]
    long = data["long"]
    user_loc = tf.constant([[lat, long]], dtype=tf.float32)
    predictions = model.predict(user_loc).numpy().tolist()[0]
    predictions = [[place_id.decode(), place_name.decode(), city.decode()] for place_id, place_name, city in predictions]
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
