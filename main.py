import argparse
import copy
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    # 深度神经网络数据加载
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.Tensor(self.label[index])
        return data, label

    def __len__(self):
        return len(self.data)


class LinearModule(nn.Module):
    # 全连接神经网络
    def __init__(self, length):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(length, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 25)
        )

    def forward(self, x):
        x = self.features(x)
        return x


def parse_args():
    # 添加训练参数
    parser = argparse.ArgumentParser(description="Add args to the model")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalization the datas")
    parser.add_argument("--method_of_dimension_reduction", type=str, default="KernelPCA",
                        help="Method of dimension reduction")
    parser.add_argument("--method_of_normalization", type=str, default="z_score",
                        help="Method of normalization")
    parser.add_argument("--method_of_feature_extraction", type=str, default="ridge",
                        help="Method of feature extraction")
    parser.add_argument("--method_of_experiment", type=str, default="leave 1 out",
                        help="Method of experiment")
    parser.add_argument("--is_all", type=bool, default=False,
                        help="Test all type of experiment")
    args = parser.parse_args()
    return args


class MinMax(object):
    # min_max归一化
    def __init__(self):
        self.max = 0
        self.min = 0

    def fit(self, data: dict):
        # 求得数据中最大最小数据
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
        # 对数据进行归一化操作
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = (data[key][i] - self.min) / (self.max - self.min)
        return data


class ZScore(object):
    # z_scire归一化
    def __init__(self):
        self.mean = 0
        self.var = 0

    def fit(self, data: dict):
        # 求得数据的平均值和方差
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        self.mean = data_process.mean()
        self.var = data_process.var()

    def process(self, data: dict):
        # 对数据进行归一化
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = (data[key][i] - self.mean) / (math.sqrt(self.var))
        return data


class Norm(object):
    # 对两种归一化操作统一接口
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
    # PCA降维
    def __init__(self):
        self.pca = PCA(n_components=0.99)

    def fit(self, data: dict):
        # 拟合PCA
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        self.pca.fit(data_process)

    def transform(self, data: dict):
        # 用PCA对数据进行降维
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


class KernelPCADimensionReduction(object):
    # KernelPCA降维
    def __init__(self):
        self.pca = KernelPCA()

    def fit(self, data: dict):
        # 拟合KernelPCA
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        self.pca.fit(data_process)

    def transform(self, data: dict):
        # 用KernelPCA对数据降维
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


class FADimensionReduction(object):
    # Factor Analysis降维
    def __init__(self):
        self.fa = FactorAnalysis(n_components=300)

    def fit(self, data: dict):
        # 拟合Factor Analysis
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        self.fa.fit(data_process)

    def transform(self, data: dict):
        # 利用Factor Analysis对数据降维
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        new_data_process = self.fa.transform(data_process)
        j = 0
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = np.array([new_data_process[j]])
                j = j + 1
        return data


class FastICAimensionReduction(object):
    # FastICA降维
    def __init__(self):
        self.fastICA = FastICA(n_components=200)

    def fit(self, data: dict):
        # 拟合fastICA
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        self.fastICA.fit(data_process)

    def transform(self, data: dict):
        # 利用fastICA对数据降维
        data_process = []
        for key in data:
            for i in range(len(data[key])):
                data_process.append(data[key][i])
        data_process = np.array(data_process)
        data_process = data_process.squeeze()
        new_data_process = self.fastICA.transform(data_process)
        j = 0
        for key in data:
            for i in range(len(data[key])):
                data[key][i] = np.array([new_data_process[j]])
                j = j + 1
        return data


class DimensionReduction(object):
    # 整合四种降维方式，统一接口
    def __init__(self, method_of_dimension_reduction: str):
        if method_of_dimension_reduction == 'PCA':
            self.dimension_reduction = PCADimensionReduction()
        elif method_of_dimension_reduction == 'KernelPCA':
            self.dimension_reduction = KernelPCADimensionReduction()
        elif method_of_dimension_reduction == 'FA':
            self.dimension_reduction = FADimensionReduction()
        elif method_of_dimension_reduction == 'fastICA':
            self.dimension_reduction = FastICAimensionReduction()

    def fit(self, data: dict):
        self.dimension_reduction.fit(data)

    def transform(self, data: dict):
        return self.dimension_reduction.transform(data)


class LinearFeatureExtraction(object):
    # 将全连接神经网络接口与回归模型接口进行统一
    def fit(self, train_data, train_label):
        # 训练模型
        self.module = LinearModule(len(train_data[0]))
        train_dataset = MyDataset(train_data, train_label)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True,
                                       num_workers=0, drop_last=False)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.module.parameters(), lr=0.001, momentum=1)
        self.module.train()
        for epoch in range(100):
            for i, data in enumerate(train_data_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.module(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, test_data):
        # 提取特征
        return np.array(self.module(torch.Tensor(np.array(test_data))).detach())


class FeatureExtraction(object):
    # 整合五种特征提取方式，统一接口
    def __init__(self, method_of_feature_extraction, word2features: dict):
        self.word2features = word2features
        if method_of_feature_extraction == 'linear_regression':
            self.model = LinearRegression()
        elif method_of_feature_extraction == 'ridge':
            self.model = Ridge(alpha=0.5)
        elif method_of_feature_extraction == 'linear_module':
            self.model = LinearFeatureExtraction()
        elif method_of_feature_extraction == 'lasso':
            self.model = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, positive=False,
                               precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
        elif method_of_feature_extraction == 'elastic_net':
            self.model = ElasticNet(alpha=0.3, l1_ratio=0.2)

    def fit(self, data: dict):
        # 训练数据
        train_data = []
        train_label = []
        for key in data:
            for i in range(len(data[key])):
                train_data.append(data[key][i].squeeze())
                train_label.append(self.word2features[key])
        self.model.fit(train_data, train_label)

    def predict(self, data: dict):
        # 提取特征
        test_data = []
        for key in data:
            for i in range(len(data[key])):
                test_data.append(data[key][i].squeeze())
        data_feature = self.model.predict(test_data)
        return data_feature


def get_train_test(data: dict, method_of_experiment: str, random_list: list):
    # 将所有数据分成训练集和测试集
    train_data = dict()
    test_data = dict()
    key_list = list(data.keys())
    for i in range(len(key_list)):
        if i in random_list:
            test_data[key_list[i]] = copy.deepcopy(data[key_list[i]])
        else:
            train_data[key_list[i]] = copy.deepcopy(data[key_list[i]])
    return train_data, test_data


def classification(feature, word2features: dict):
    # 根据提取的特征进行分类，利用欧式距离，找出最接近的标签
    min_distance = np.linalg.norm(feature - word2features[list(word2features.keys())[0]])
    min_key = list(word2features.keys())[0]
    for key in word2features:
        distance = np.linalg.norm(feature - word2features[key])
        if distance < min_distance:
            min_distance = distance
            min_key = key
    return min_key


def test(test_data: dict, feature_extraction: FeatureExtraction, word2features: dict, method_of_experiment: str):
    # 对训练好的模型进行测试，返回准确率
    right_num = 0
    total_num = 0
    test_feature = feature_extraction.predict(test_data)
    if method_of_experiment == 'leave 2 out':
        part_of_word2features = dict()
        key_list = list(test_data.keys())
        for key in key_list:
            part_of_word2features[key] = word2features[key]
        j = 0
        for key in test_data:
            for i in range(len(test_data[key])):
                feature_key = classification(test_feature[j], part_of_word2features)
                j = j + 1
                total_num = total_num + 1
                if feature_key == key:
                    right_num = right_num + 1
        acc = (right_num / total_num) * 100
    elif method_of_experiment == 'leave 1 out':
        j = 0
        for key in test_data:
            for i in range(len(test_data[key])):
                feature_key = classification(test_feature[j], word2features)
                j = j + 1
                total_num = total_num + 1
                if feature_key == key:
                    right_num = right_num + 1
        acc = (right_num / total_num) * 100
    return acc


def train_and_test(original_data: dict, word2features: dict, verbs: list, normalize: bool,
                   method_of_dimension_reduction: str,
                   method_of_normalization: str, method_of_feature_extraction: str, method_of_experiment: str,
                   random_list: list):
    # 对模型进行训练并对训练好的模型进行测试
    data = copy.deepcopy(original_data)
    # 获取测试数据和训练数据
    train_data, test_data = get_train_test(data, method_of_experiment, random_list)
    # 数据降维
    dimension_reduction = DimensionReduction(method_of_dimension_reduction)
    dimension_reduction.fit(train_data)
    train_data = dimension_reduction.transform(train_data)
    test_data = dimension_reduction.transform(test_data)
    # 数据归一化
    if normalize:
        normalization = Norm(method_of_normalization)
        normalization.fit(train_data)
        train_data = normalization.norm(train_data)
        test_data = normalization.norm(test_data)
    # 训练模型
    feature_extraction = FeatureExtraction(method_of_feature_extraction, word2features)
    feature_extraction.fit(train_data)
    # 测试模型
    acc = test(test_data, feature_extraction, word2features, method_of_experiment)
    return acc


if __name__ == '__main__':
    # 添加运行参数
    args = parse_args()
    normalize = args.normalize
    method_of_dimension_reduction = args.method_of_dimension_reduction
    method_of_normalization = args.method_of_normalization
    method_of_feature_extraction = args.method_of_feature_extraction
    method_of_experiment = args.method_of_experiment
    is_all = args.is_all
    # 加载数据
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
    # 训练并测试
    accs = []
    if is_all:
        process_bar = tqdm(range(int(len(datas) * len(datas[0]) * (len(datas[0]) - 1) / 2)))
        if method_of_experiment == 'leave 2 out':
            for i in range(len(datas)):
                for k in range(len(datas[i])):
                    j = k + 1
                    while j < len(datas[i]):
                        acc = train_and_test(datas[i], word2features, verbs, normalize, method_of_dimension_reduction,
                                             method_of_normalization, method_of_feature_extraction,
                                             method_of_experiment,
                                             [k, j])
                        accs.append(acc)
                        j = j + 1
                        process_bar.update(1)
        if method_of_experiment == 'leave 1 out':
            for i in tqdm(range(len(datas))):
                for k in range(len(datas[i])):
                    acc = train_and_test(datas[i], word2features, verbs, normalize, method_of_dimension_reduction,
                                         method_of_normalization, method_of_feature_extraction, method_of_experiment,
                                         [k])
                    accs.append(acc)
    else:
        if method_of_experiment == 'leave 2 out':
            for i in range(len(datas)):
                for k in range(10):
                    random1 = random.randint(0, 59)
                    random2 = random.randint(0, 59)
                    while random2 == random1:
                        random2 = random.randint(0, 59)
                    acc = train_and_test(datas[i], word2features, verbs, normalize, method_of_dimension_reduction,
                                         method_of_normalization, method_of_feature_extraction,
                                         method_of_experiment,
                                         [random1, random2])
                    accs.append(acc)
        if method_of_experiment == 'leave 1 out':
            random_num = random.randint(0, 59)
            for i in tqdm(range(len(datas))):
                for k in range(90):
                    acc = train_and_test(datas[i], word2features, verbs, normalize, method_of_dimension_reduction,
                                         method_of_normalization, method_of_feature_extraction, method_of_experiment,
                                         [random_num])
                    accs.append(acc)
    average_accuracy = np.array(accs).sum() / len(accs)
    print('average accuracy:' + str(average_accuracy) + '%')
