# -*- encoding:utf-8 -*-
import numpy as np
import os
import glob
from skimage.io import imread
from skimage.transform import resize

from utils import save_pickle

labels = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking']
np.random.seed(0)


def generate_dataset(img_root_path, split, save_path):
    paths = os.listdir(img_root_path)
    imgs = []
    onehot_vectors = []
    for path in paths:
        if os.path.isdir(os.path.join(img_root_path, path)):
            label = path
            onehot_vector = np.zeros([len(labels)]).astype(np.float32)
            onehot_vector[labels.index(label)] = 1
            _imgs = glob.glob(os.path.join(img_root_path, path)+"/*.jpg")
            for _img in _imgs:
                imgs.append(_img)
                onehot_vectors.append(onehot_vector)

    sample_list = np.arange(len(imgs))
    print(len(imgs), len(onehot_vectors))
    np.random.shuffle(sample_list)
    if split == "train":
        img_matrix = np.zeros([len(sample_list[:int(len(imgs) * 0.75)]), 224, 224, 3]).astype(np.float32)
        label_matrix = np.zeros([len(sample_list[:int(len(imgs) * 0.75)]), len(labels)]).astype(np.float32)
        for k, i in enumerate(sample_list[:int(len(imgs) * 0.75)]):
            img_file = imgs[i]
            img = imread(img_file)
            img = resize(img, (224, 224))
            img_matrix[k] = img
            label_matrix[k] = onehot_vectors[i]
    elif split == "val":
        img_matrix = np.zeros([len(sample_list[int(len(imgs) * 0.75):]), 224, 224, 3]).astype(np.float32)
        label_matrix = np.zeros([len(sample_list[int(len(imgs) * 0.75):]), len(labels)]).astype(np.float32)
        for k, i in enumerate(sample_list[int(len(imgs) * 0.75):]):
            img_file = imgs[i]
            img = imread(img_file)
            img = resize(img, (224, 224))
            img_matrix[k] = img
            label_matrix[k] = onehot_vectors[i]
    else:
        return
    img_matrix = img_matrix / 255.0
    save_imgs = save_path + "/" + split + ".imgs.pkl"
    save_labels = save_path + "/" + split + ".labels.pkl"
    save_pickle(img_matrix, save_imgs)
    save_pickle(label_matrix, save_labels)