import numpy as np
import random
import math
import os
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pathlib import Path

# PATH= Path("../mesh_data/ModelNet10")
# PATH= Path("../mesh_data/ModelNet40")
class RandRotation_z(object):
    def __init__(self, with_normal=False, SO3=False):
        self.with_normal = with_normal
        self.SO3 = SO3

    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        roll, pitch, yaw = np.random.rand(3)*np.pi*2
        if self.SO3 is False:
            pitch, roll = 0.0, 0.0

        rot_matrix = R.from_euler('XZY', (roll, yaw, pitch)).as_matrix()

        # Transform the rotation matrix for points with normals. Shape (6,6)
        zero_matrix = np.zeros((3,3))
        tmp_matrix = np.concatenate((rot_matrix,zero_matrix),axis=1) # [R,0]
        tmp_matrix_2 = np.concatenate((zero_matrix, rot_matrix), axis=1) # [0,R]
        # [[R,0],[0,R]]
        rot_matrix_with_normal = np.concatenate((tmp_matrix, tmp_matrix_2), axis=0)
        if self.with_normal is True:
            rot_pointcloud = rot_matrix_with_normal.dot(pointcloud.T).T
        else:
            rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


class PointSampler(object):
    '''
    Uniformly sample the faces. Then randomly sample the points on the faces.
    '''
    def __init__(self, output_size, with_normal=False):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.with_normal = with_normal

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))
        sampled_points_with_norm = np.zeros((self.output_size, 6)) # points with normals
        points_normal = np.zeros((self.output_size, 3)) # normals

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

            # Cross production AB * AC
            points_normal[i] = np.cross(
                (verts[sampled_faces[i][1]]-verts[sampled_faces[i][0]]),
                (verts[sampled_faces[i][2]]-verts[sampled_faces[i][0]])
            )
            points_normal[i] = points_normal[i]/np.linalg.norm(points_normal[i])

        sampled_points_with_norm = np.concatenate((sampled_points,points_normal), axis=1)
        # print(sampled_points_with_norm.shape)

        if self.with_normal is True:
            # print(sampled_points_with_norm)
            return sampled_points_with_norm
        else:
            # print(sample_point)
            return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        if pointcloud.shape[1] == 3: # without normals
            norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) # translate to origin
            norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1)) # normalize
            return norm_pointcloud

        else: # with normals
            pointcloud_tmp, pointcloud_norm = np.split(pointcloud, 2, axis=1)
            # translate points to origin
            norm_pointcloud_tmp = pointcloud_tmp - np.mean(pointcloud_tmp, axis=0)
            # normalize points
            norm_pointcloud_tmp /= np.max(np.linalg.norm(norm_pointcloud_tmp, axis=1))
            norm_pointcloud = np.concatenate((norm_pointcloud_tmp, pointcloud_norm), axis=1)
            return  norm_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([PointSampler(1024),
                               Normalize(),
                               ToTensor()
                               ])

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


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        # self.transforms = transform if not valid else default_transforms()
        self.transforms = transform
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}




if __name__ == "__main__":

    train_transforms = transforms.Compose([
            PointSampler(1024, with_normal=True),
            Normalize(),
            RandRotation_z(),
            RandomNoise(),
            ToTensor()
            ])
    test_transforms = transforms.Compose([
            PointSampler(50, with_normal=True),
            Normalize(),
            RandRotation_z(with_normal=True, SO3=True),
            RandomNoise(),
            ToTensor()
            ])

    test_ds = PointCloudData(PATH, valid=True, folder='test', transform=test_transforms)
    test_pointcloud = test_ds.__getitem__(10)
    # print(test_pointcloud.__sizeof__())
    # print(test_pointcloud)
