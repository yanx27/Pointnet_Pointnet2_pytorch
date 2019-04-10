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
        return train_data,train_label,train_Seglabel, test_data,test_label, test_Seglabel


class ShapeNetDataLoader(Dataset):
    def __init__(self, data, labels,seg_label, npoints=1024, jitter=False, rotation=False, normalize = False):
        self.data = data
        self.labels = labels
        self.seg_label = seg_label
        self.npoints = npoints
        self.rotation = rotation
        self.jitter = jitter
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def pc_augment_to_point_num(self, pts, pn):
        assert (pts.shape[0] <= pn)
        cur_len = pts.shape[0]
        res = np.array(pts)
        while cur_len < pn:
            res = np.concatenate((res, pts))
            cur_len += pts.shape[0]
        return res[:pn]


    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]
        seg = self.seg_label[index]
        #print(point_set.shape, seg.shape)

        if self.npoints<=point_set.shape[0]:
            choice = np.random.choice(len(seg), self.npoints, replace=True) #resample
            point_set = point_set[choice, :]
            seg = seg[choice]
            point_set_norm = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set_norm = point_set_norm / dist  # scale
            if self.jitter:
                point_set += np.random.normal(0, 0.02, size=point_set.shape)


        else:
            point_set_norm = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set_norm = point_set_norm / dist  # scale
            point_set = self.pc_augment_to_point_num(point_set,self.npoints)
            seg = self.pc_augment_to_point_num(seg,self.npoints)
            if self.jitter:
                point_set += np.random.normal(0, 0.02, size=point_set.shape)

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        if self.normalize:
            return point_set_norm, labels, seg, point_set_norm
        else:
            return point_set,labels,seg,point_set_norm



