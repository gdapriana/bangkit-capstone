from collections import defaultdict
import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt


def load_data():
    # Train Data
    item_train = genfromtxt('./dataset/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./dataset/content_user_train.csv', delimiter=',')
    y_train = genfromtxt('./dataset/content_y_train.csv', delimiter=',')
    item_vecs = genfromtxt('./dataset/content_item_vecs.csv', delimiter=',')

    # Features
    with open('./dataset/content_item_train_header.txt', newline='') as f:
        item_features = list(csv.reader(f))[0]
    with open('./dataset/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]

    # Destination Dictionary
    count = 0
    destination_dict = defaultdict(dict)
    with open('./dataset/content_destination_list.csv', newline='') as csvf:
        reader = csv.reader(csvf, delimiter=',', quotechar='"')
        for row in reader:
            if count == 0:
                count += 1
            else:
                count += 1
                destination_id = int(row[0])
                destination_dict[destination_id]['name'] = row[1]
                destination_dict[destination_id]['category'] = row[2]
    return item_train, user_train, y_train, item_features, user_features, item_vecs, destination_dict


def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):
    if user:
        flist = [".0f", ".0f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f",
                 ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f",
                 ".0f", ".0f", ".0f"]

    head = features[:vs]
    if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    hdr = head + genres
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0, x_train.shape[0]):
        if count == maxcount: break
        count += 1
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)
                     ])
    return pd.DataFrame(disp)


def split_str(ifeatures, smax):
    ofeatures = []
    for s in ifeatures:
        if not ' ' in s:
            if len(s) > smax:
                mid = int(len(s) / 2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return ofeatures


def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs


def print_pred_destination(y_p, item, destination_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    disp = [["y_p", "id", "rating ave", "name", "category"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        destination_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     destination_dict[destination_id]['name'], destination_dict[destination_id]['category']])

    return pd.DataFrame(disp)


if __name__ == '__main__':
    load_data()
