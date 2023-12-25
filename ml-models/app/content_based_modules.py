import json
import pandas as pd
from numpy import genfromtxt
import csv
import tabulate
from collections import defaultdict
import numpy as np


def load_data():
    item_train = genfromtxt('data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('data/content_user_train.csv', delimiter=',')
    y_train = genfromtxt('data/content_y_train.csv', delimiter=',')
    with open('data/content_item_train_header.txt', newline='') as f:
        item_features = list(csv.reader(f))[0]
    with open('data/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('data/content_item_vecs.csv', delimiter=',')

    destination_dict = defaultdict(dict)
    count = 0

    with open('data/content_destination_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1
            else:
                count += 1
                destination_id = int(line[0])
                destination_dict[destination_id]["name"] = line[1]
                destination_dict[destination_id]["category"] = line[2]

    return item_vecs, destination_dict, item_vecs, item_train, user_train, y_train


def split_str(ifeatures, smax):
    ofeatures = []
    for s in ifeatures:
        if not ' ' in s:  # skip string that already have a space
            if len(s) > smax:
                mid = int(len(s) / 2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return ofeatures


def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):
    if user:
        flist = [".0f", ".0f", ".1f",
                 ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f",
                 ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

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
    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
    return table


def gen_user_vecs(user_vec, num_items):
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs


def get_predict(y_p, item, destination_dict, maxcount=10):
    count = 0
    disp = [["y_p", "place id", "rating ave", "name", "category"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        destination_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     destination_dict[destination_id]['name'], destination_dict[destination_id]['category']])

    # table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
    data = convert_to_json(disp)
    return data


def convert_to_json(data):
    json_data = []
    for row in data:
        data_dict = {
            "y_p": str(row[0]),
            "id": str(row[1]),
            "rating_ave": str(row[2]),
            "name": str(row[3]),
            "category": str(row[4])
        }
        json_data.append(data_dict)
    json_data.pop(0)
    return json_data
