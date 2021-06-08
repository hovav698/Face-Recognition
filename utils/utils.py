import shutil
import os
import pathlib
from tensorflow.keras.utils import get_file
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def get_data():
    data_dir = pathlib.Path('data/yalefaces.zip')
    if not data_dir.exists():
        get_file(
            'yalefaces.zip',
            origin="http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip",
            extract=True,
            cache_dir='.', cache_subdir='data'
        )
        os.remove('data/yalefaces/Readme.txt')
        os.remove('data/yalefaces.zip')


# resize and normal images
def load_and_reformat_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((60, 80)),
        transforms.ToTensor()
    ])

    img = transform(img).numpy()
    return img


# the images will be resized to H,W
H = 60
W = 80


# load images and labels
def load_data(filenames):
    n = len(filenames)
    images = np.empty((n, 1, H, W,), dtype=np.float32)  # batch_size x 1 x image height x image width
    labels = np.empty(n, dtype=np.float32)

    for i, f in enumerate(filenames):
        img = load_and_reformat_image(f)
        images[i] = img
        f = f.split('/', 1)[-1]
        f = f.split('subject', 1)[1]
        f = f.split('.')[0]
        label = np.float32(f)
        labels[i] = label - 1

    return images, labels


# the dataset that we will run the model on will contain pairs of images,
# positive targets pairs means the two images are of the same person.
# negative targets pair means the two images are of different person.
# we don't want to save all the image pairs in the memory becuase it will be redundant and take too much space
# therefor we will first get the indexs of the of the positive and negative target pairs

def get_target_pairs(labels):
    n_images = len(labels)
    neg_target_pairs = []
    pos_target_pairs = []

    for i in range(n_images):
        for j in range(n_images):
            # for the negative pairs we dont want images of the same person, the pair labels must be different
            if ((j, i) not in neg_target_pairs and labels[i] != labels[j]):
                neg_target_pairs.append((i, j))

            # for the positive pairs we  want images of the same person, the pair labels must be equal
            # in case i!=j we dont want both of the indexes (i,j) and (j,i)
            # we also dont want to compare an image to itself, so i!=j
            if (j != i and (j, i) not in pos_target_pairs and labels[i] == labels[j]):
                pos_target_pairs.append((i, j))

    neg_target_pairs = np.array(neg_target_pairs)
    pos_target_pairs = np.array(pos_target_pairs)

    return pos_target_pairs, neg_target_pairs
