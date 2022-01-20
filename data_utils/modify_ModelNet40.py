import numpy as np
import random
import math
import os
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pathlib import Path


def read_off(file):
    head = file.readline().strip() # this is the first line of the file
    if 'OFF' != head:
        # print("1111111")
        # print(file)
        first_line = head.strip("OFF")
        n_verts, n_faces, __ = tuple([int(s) for s in first_line.strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    print(n_verts, n_faces)
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def modify(folder="train"):
    root_dir = Path("../mesh_data/ModelNet40")
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print(classes)
    files = []

    for category in classes.keys():
        new_dir = root_dir/Path(category)/folder
        for file in os.listdir(new_dir):
            if file.endswith('.off'):
                sample = {}
                sample['pcd_path'] = new_dir/file
                sample['category'] = category
                files.append(sample)

    print(len(files))




if __name__ == "__main__":
    # modify()
    with open("../mesh_data/ModelNet40/desk/train/desk_0002.off", 'r') as f:
        read_off(f)
