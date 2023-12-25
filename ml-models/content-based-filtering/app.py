from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from modules import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')


@app.route('/category_predict', methods=['POST'])
def genre_predict():
    new_data = request.get_json()
    new_id = new_data["id"]
    new_rating_count = new_data["rating_count"]
    new_rating_ave = new_data["rating_ave"]
    new_bahari = new_data["bahari"]
    new_budaya = new_data["budaya"]
    new_cagar_alam = new_data["cagar_alam"]
    new_pusat_perbelanjaan = new_data["pusat_perbelanjaan"]
    new_taman_hiburan = new_data["taman_hiburan"]
    new_tempat_ibadah = new_data["tempat_ibadah"]
    user_vec = np.array([[new_id, new_rating_count, new_rating_ave, new_bahari, new_budaya, new_cagar_alam, new_pusat_perbelanjaan, new_taman_hiburan, new_tempat_ibadah]])

    item_vecs, destination_dict, item_vecs, item_train, user_train, y_train = load_data()
    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))

    u_s = 3
    i_s = 1

    user_vecs = gen_user_vecs(user_vec, len(item_vecs))
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])
    y_pu = scalerTarget.inverse_transform(y_p)

    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    predicts = get_predict(sorted_ypu, sorted_items, destination_dict, maxcount=10)
    return predicts


if __name__ == '__main__':
    app.run(port=5000, debug=True)