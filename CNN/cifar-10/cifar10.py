import pickle

with open('./dataset/data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    print(dict)
