import pickle
from urllib import request
import tensorflow as tf
import pandas as pd


def load_data():
    file_name = "dataset.csv"
    request.urlretrieve("https://raw.githubusercontent.com/Touventure/models/main/data/tourism_with_id.csv",
                        filename=file_name)
    df = pd.read_csv(file_name)
    df = df[["Place_Id", "Place_Name", "City", "Lat", "Long"]].rename(
        columns={"Place_Id": "place_id", "Place_Name": "place_name", "City": "city", "Lat": "lat", "Long": "long"})
    df['place_id'] = df['place_id'].apply(str)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


class KNN:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X, K=10):
        r = tf.reduce_sum(X ** 2, axis=1) - 2 * tf.matmul(X, tf.transpose(self.X)) + tf.reduce_sum(self.X ** 2,
                                                                                                   axis=1)
        distances = tf.sqrt(r)
        _, indices = tf.nn.top_k(tf.negative(distances), k=K)
        return tf.gather(self.y, indices)


def model_init():
    dataset = load_data()
    tensor = tf.convert_to_tensor(dataset[['lat', 'long']].values, dtype=tf.float32)
    model = KNN(tensor, dataset[['place_id', 'place_name', 'city']])
    return model


if __name__ == "__main__":
    model = model_init()
    pickle.dump(model, open("model.pkl", "wb"))
