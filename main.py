import argparse
import os
import pickle
import random

import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def parse_args():
    parser = argparse.ArgumentParser(description="Add args to the model")
    parser.add_argument("--normalize", type=bool, default=False, help="Normalization the datas")
    parser.add_argument("--method_of_dimension_reduction", type=str, default="PCA",
                        help="Method of dimension reduction")
    parser.add_argument("--method_of_normalization", type=str, default="min_max",
                        help="Method of normalization")
    parser.add_argument("--method_of_feature_extraction", type=str, default="linear_regression",
                        help="Method of feature extraction")
    parser.add_argument("--method_of_experiment", type=str, default="leave 2 out",
                        help="Method of experiment")
    args = parser.parse_args()
    return args


class MinMax(object):
    def fit(self, data: dict):
        self.min = data[list(data.keys())[0]][0].min()
        self.max = data[list(data.keys())[0]][0].max()
        for key in data:
            for i in range(len(data[key])):
                data_max = data[key][i].max()
                data_min = data[key][i].min()
                if data_max > self.max:
                    self.max = data_max
                if data_min < self.min:
                    self.min = data_min

    def process(self, data: dict):
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = (data[key][i] - self.min) / (self.max - self.min)
        return data


class ZScore(object):
    def fit(self, data: dict):
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        self.mean = data_process.mean()
        self.var = data_process.var()

    def process(self, data: dict):
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = (data[key][i] - self.mean) / (math.sqrt(self.var))
        return data


class Norm(object):
    def __init__(self, method_of_normalization: str):
        if method_of_normalization == 'min_max':
            self.normalization = MinMax()
        elif method_of_normalization == 'z_score':
            self.normalization = ZScore()

    def fit(self, data: dict):
        self.normalization.fit(data)

    def norm(self, data: dict):
        return self.normalization.process(data)


class PCADimensionReduction(object):
    def __init__(self):
        self.pca = PCA(n_components=0.99)

    def fit(self, data: dict):
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        self.pca.fit(data_process)

    def transform(self, data: dict):
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        new_data_process = self.pca.transform(data_process)
        j = 0
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = np.array([new_data_process[j]])
                j = j + 1
        return data


class DimensionReduction(object):
    def __init__(self, method_of_dimension_reduction: str):
        if method_of_dimension_reduction == 'PCA':
            self.dimension_reduction = PCADimensionReduction()

    def fit(self, data: dict):
        self.dimension_reduction.fit(data)

    def transform(self, data: dict):
        return self.dimension_reduction.transform(data)


class FeatureExtraction(object):
    def __init__(self, method_of_feature_extraction, word2features: dict):
        self.word2features = word2features
        if method_of_feature_extraction == 'linear_regression':
            self.model = LinearRegression()

    def fit(self, data: dict):
        train_data = []
        train_label = []
        for key in data:
            for i in range(len(data[key])):
                train_data.append(data[key][i].squeeze())
                train_label.append(self.word2features[key])
        self.model.fit(train_data, train_label)

    def predict(self, data: dict):
        test_data = []
        for key in data:
            for i in range(len(data[key])):
                test_data.append(data[key][i].squeeze())


def get_train_test(data: dict, method_of_experiment: str):
    if method_of_experiment == 'leave 2 out':
        random1 = random.randint(0, 59)
        random2 = random.randint(0, 59)
        while random1 == random2:
            random2 = random.randint(0, 59)
        random_list = [random1, random2]
    elif method_of_experiment == 'leave 1 out':
        random1 = random.randint(0, 59)
        random_list = [random1]
    train_data = dict()
    test_data = dict()
    key_list = list(data.keys())
    for i in range(len(key_list)):
        if i in random_list:
            test_data[key_list[i]] = data[key_list[i]]
        else:
            train_data[key_list[i]] = data[key_list[i]]
    return train_data, test_data


def train_and_test(data: dict, word2features: dict, verbs: list, normalize: bool, method_of_dimension_reduction: str,
                   method_of_normalization: str, method_of_feature_extraction: str, method_of_experiment: str):
    train_data, test_data = get_train_test(data, method_of_experiment)
    if normalize:
        normalization = Norm(method_of_normalization)
        normalization.fit(train_data)
        train_data = normalization.norm(train_data)
        test_data = normalization.norm(test_data)
    dimension_reduction = DimensionReduction(method_of_dimension_reduction)
    dimension_reduction.fit(train_data)
    train_data = dimension_reduction.transform(train_data)
    test_data = dimension_reduction.transform(test_data)
    feature_extraction = FeatureExtraction(method_of_feature_extraction, word2features)
    feature_extraction.fit(train_data)
    return 1


if __name__ == '__main__':
    args = parse_args()
    normalize = args.normalize
    method_of_dimension_reduction = args.method_of_dimension_reduction
    method_of_normalization = args.method_of_normalization
    method_of_feature_extraction = args.method_of_feature_extraction
    method_of_experiment = args.method_of_experiment
    datas_dir = './data/data_science'
    verbs_dir = './data/verbs.pkl'
    word2features_dir = './data/word2features.pkl'
    verbs_f_read = open(verbs_dir, 'rb')
    verbs = pickle.load(verbs_f_read)
    word2features_f_read = open(word2features_dir, 'rb')
    word2features = pickle.load(word2features_f_read)
    data_files = os.listdir(datas_dir)
    datas = []
    for data_file in data_files:
        data_dir = datas_dir + '/' + data_file
        data_f_read = open(data_dir, 'rb')
        data = pickle.load(data_f_read)
        datas.append(data)
    accs = []
    for i in range(len(datas)):
        acc = train_and_test(datas[i], word2features, verbs, normalize, method_of_dimension_reduction,
                             method_of_normalization, method_of_feature_extraction, method_of_experiment)
