from astromcad.scripts.utils.utils import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data():
    target = load("../../data/target_cls")
    x_data = load("../../data/x_data")
    host_galaxy_info = load("../../data/host_galaxy_info")
    return x_data, host_galaxy_info, target


def cut_by_class(x_data, host_galaxy_info, target, limit = 13000):
    valid = [False] * len(target)
    for class_ in np.unique(target):
        cnt = 0
        for i in range(len(target)):
            if (target[i] == class_):
                valid[i] = True
                cnt+=1
            if (cnt == limit): # cut down
                break
    
    for i in range(len(target) - 1, -1, -1):
        if not valid[i]:
            del target[i]
            del x_data[i]
            del host_galaxy_info[i]

def max_length(x_data):
    lengths = []
    for lc in x_data:
        lengths.append(len(lc))
    ntimesteps = np.max(lengths)
    return ntimesteps

def dilate(scaled_time, redshift):
    time = ((scaled_time * 100) - 30)
    corrected_time = (time / (1 + redshift))
    return (corrected_time + 30) / 100 


def apply_dilation(x_data, host_galaxy_info):
    for ind in range(len(x_data)):
        x_data[ind][:, 1] = dilate(x_data[ind][:, 1], host_galaxy_info[ind][0])

def pad_curves(x_data, ntimesteps):
    for ind in range(len(x_data)):
        x_data[ind] = np.pad(x_data[ind], ((0, ntimesteps - len(x_data[ind])), (0, 0)))

def split_data(target, x_data, host_galaxy_info):
    y_data_anom = []
    y_data_norm = []
    x_data_norm = []
    x_data_anom = []
    host_gal_anom = []
    host_gal = []

    for i in range(len(target)):

        if (target[i] in anom_classes):
            x_data_anom.append(x_data[i])
            y_data_anom.append(target[i])
            host_gal_anom.append(host_galaxy_info[i])

        else:
            x_data_norm.append(x_data[i])
            y_data_norm.append(target[i])
            host_gal.append(host_galaxy_info[i])

    return (x_data_norm, host_gal, y_data_norm), (x_data_anom, host_gal_anom, y_data_anom)

def generate_class_weights(y_train):
    class_weights = {i : 0 for i in range(y_train.shape[1])}

    for value in y_train:
        class_weights[np.argmax(value)]+=1

    for id in class_weights.keys():
        class_weights[id] = len(y_train) / class_weights[id]
        return class_weights


def get_data():
    print("Loading data...")
    x_data, host_galaxy_info, target = load_data()

    cut_by_class(x_data, host_galaxy_info, target)
    print("Classes Available: ", list(np.unique(target)))

    ntimesteps = max_length(x_data)
    print("Max Observations: ", ntimesteps)
    
    print("Preprocessing data...")
    # apply_dilation(x_data, host_galaxy_info)
    pad_curves(x_data, ntimesteps)
    
    (x_data_norm, host_gal, y_data_norm), (x_data_anom, host_gal_anom, y_data_anom) = split_data(target, x_data, host_galaxy_info)
    
    enc = OneHotEncoder(handle_unknown='ignore')
    y_data_norm = enc.fit_transform(np.array(y_data_norm).reshape(-1, 1)).todense()
    create_ordered_class_names(enc)    

    X_train, X_test, host_gal_train, host_gal_test, y_train, y_test = train_test_split(x_data_norm, host_gal, y_data_norm, random_state = 40, test_size = 0.1)
    X_train, X_val, host_gal_train, host_gal_val, y_train, y_val = train_test_split(X_train, host_gal_train, y_train, random_state = 40, test_size = 1/9)

    class_weights = generate_class_weights(y_train)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    host_gal_train = np.array(host_gal_train)
    host_gal_test = np.array(host_gal_test)
    host_gal_val = np.array(host_gal_val)

    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)
    
    x_data_anom = np.array(x_data_anom)
    host_gal_anom = np.array(host_gal_anom)
    y_data_anom = np.squeeze(y_data_anom)
    return X_train, X_val, X_test, host_gal_train, host_gal_val, host_gal_test, y_train, y_val, y_test, class_weights, ntimesteps, x_data_anom, host_gal_anom, y_data_anom  