# -*- encoding:utf-8 -*-
import pickle
from skimage.io import imread, imsave
from skimage.transform import resize
import glob


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=2)
        print('Saved %s..' % path)


def resize_image(input_path, save_path, in_file_type='tif', save_file_type='tif'):
    images = glob.glob(input_path+"/*."+in_file_type)
    for image in images:
        file_name =image[image.rindex("\\")+1:]
        img = imread(image)
        new_img = resize(img, (224, 224))
        imsave(save_path+"/"+file_name.replace(in_file_type, save_file_type), new_img)
