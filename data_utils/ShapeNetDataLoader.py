# *_*coding:utf-8 *_*
import numpy as np
import h5py
import torch
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def load_data(dir, classification = False):
    data_train0, label_train0, Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1, Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2, Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3, Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4, Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_train5, label_train5, Seglabel_train5 = load_h5(dir + 'ply_data_train5.h5')
    data_test0, label_test0, Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1 ,label_test1, Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4,data_train5])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4,label_train5])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4,Seglabel_train5])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel


class ShapeNetDataLoader(Dataset):
    def __init__(self, data, labels, npoints=1024, data_augmentation=True):
        self.data = data
        self.labels = labels
        self.npoints = npoints
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point_set = self.data[index]
        seg = self.labels[index]
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        return point_set, seg



