import os
import sys
import time
import glob
import json
import scipy
import torch
import pickle
import shutil
import trimesh
import argparse

import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# set released camera ids
RELEASE_CAMERAS = ['0004', '0028', '0052', '0076']
# set surface label and color: skin-0, hair-1, shoe-2, upper-3, lower-4, outer-5
SURFACE_LABEL = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']
SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])


# # ------------------------ Data Utilities ------------------------ # #

# load data from pkl
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# save data to pkl
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# show RGB image using plt, {q, s}
def show_image(img, name='image'):
    fig = plt.figure(figsize=(32, 18))
    fig.suptitle(name)
    plt.imshow(img)
    plt.show()


# create normalized grid with n_per_side
def create_grid(n_per_side, sx=1, sy=1):
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x * sx, points_y * sy], axis=-1).reshape(-1, 2)
    return points

# draw line with start(hs, ws) and end(he, we)
def draw_line(img, start, end, color=(0, 0, 0), thickness=1):
    cv.line(img, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), color=color, thickness=thickness)
    return img

# draw text with position(h, w)
def draw_text(img, text, position, font_scale=1.5, color=(0, 0, 0), thickness=2):
    cv.putText(img, text, (int(position[1]), int(position[0])), fontFace=1, fontScale=font_scale, color=color, thickness=thickness, lineType=cv.LINE_AA)
    return img

# draw point(h, w)
def draw_point(img, point, radius=2, color=(0, 0, 0), thickness=-1):
    cv.circle(img, (int(point[1]), int(point[0])), radius=radius, color=color, thickness=thickness)

# draw rectangle with start(hs, ws) and end(he, we)
def draw_rectangle(img, start, end, color=(0, 0, 0), thickness=1):
    cv.rectangle(img, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), color=color, thickness=thickness)
    return img

# draw segmentation masks
def draw_masks(masks):
    # init img_masks
    img_masks = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 4))
    img_masks[:, :, 3] = 0
    # paint mask to img
    for mask in masks:
        img_masks[mask['segmentation']] = np.concatenate([np.random.random(3), [0.6]])
    return (img_masks * 255).astype(np.uint8)
