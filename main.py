import os
import pickle

if __name__ == '__main__':
    datas_dir = './data/data_science'
    verbs_dir = './data/verbs.pkl'
    word2features_dir = './data/word2features.pkl'
    verbs_f_read = open(verbs_dir, 'rb')
    verbs = pickle.load(verbs_f_read)  # 25
    word2features_f_read = open(word2features_dir, 'rb')
    word2features = pickle.load(word2features_f_read)  # 60×25
    '''print(word2features)
    print(verbs)'''
    data_files = os.listdir(datas_dir)
    # print(data_files)
    datas = []  # 9×60×6×1×21764
    for data_file in data_files:
        data_dir = datas_dir + '/' + data_file
        data_f_read = open(data_dir, 'rb')
        data = pickle.load(data_f_read)
        datas.append(data)
    print(datas[0]['refrigerator'][0].shape)
