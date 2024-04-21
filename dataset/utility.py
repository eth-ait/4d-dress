import os
import glob
import tqdm
import torch
import pickle
import trimesh
import argparse
import cv2 as cv
import numpy as np
from PIL import Image

# TODO: Set 4D-DRESS Dataset Folder Here
DATASET_DIR = "Please set 4D-DRESS DATASET_DIR in utility.py"

# set released camera ids
RELEASE_CAMERAS = ['0004', '0028', '0052', '0076']
# set surface label and color: skin-0, hair-1, shoe-2, upper-3, lower-4, outer-5
SURFACE_LABEL = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']
SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])


# load data from pkl_dir
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# save data to pkl_dir
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# get xyz rotation matrix
def rotation_matrix(angle, axis='x'):
    # get cos and sin from angle
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # get totation matrix
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R