# *_*coding:utf-8 *_*
import numpy as np
import os
import h5py
import warnings
from pyntcloud import PyntCloud
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import show_point_cloud
warnings.filterwarnings('ignore')

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def load_segdata(dir):
    data_train0, _, label_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, _, label_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, _, label_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, _, label_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, _, label_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_train5, _, label_train5 = load_h5(dir + 'ply_data_train5.h5')
    data_test0, _, label_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1 ,_, label_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4,data_train5])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4,label_train5])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    print('The shape of training data is:',train_data.shape)
    print('The shape of training label is:',train_label.shape)
    print('The shape of test data is:',test_data.shape)
    print('The shape of test label is:',test_label.shape)

    return train_data, train_label,test_data,test_label

class ShapeNetDataLoader(Dataset):
    def __init__(self, data, labels, rotation=None):
        self.data = data
        self.labels = labels
        self.rotation = rotation

    def __len__(self):
        return len(self.data)

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data

    def __getitem__(self, index):

        if self.rotation is not None:
            pointcloud = self.data[index]
            angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)

            return pointcloud, self.labels[index]
        else:
            return self.data[index], self.labels[index]


