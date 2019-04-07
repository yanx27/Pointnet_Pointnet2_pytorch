import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, rotation = None):
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