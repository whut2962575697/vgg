# -*- encoding:utf-8 -*-
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file
