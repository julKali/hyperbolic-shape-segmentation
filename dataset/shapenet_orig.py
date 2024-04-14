import os, os.path as osp
import glob
import h5py
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

url = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
BASE_DIR = osp.dirname(os.path.abspath(__file__))
DATA_DIR = osp.join(BASE_DIR, 'data')


def download_shapenetpart():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        zipfile = os.path.basename(url)
        os.system('wget --no-check-certificate %s; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5' % partition))
    for h5_name in file:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def jitter(x: np.ndarray, sigma=0.001, clip=0.005, p=0.5):
    if np.random.rand() < p:
        return x + np.clip(sigma * np.random.randn(*x.shape), -clip, clip)
    else:
        return x


def rotate(x: np.ndarray, axis='y', angle=15, p=0.5):
    # x: (n, 3)
    # rotate along y-axis (up)
    if np.random.rand() < p:
        R = Rotation.from_euler(axis, np.random.uniform(-angle, angle), degrees=True).as_matrix()
        x = x @ R
        return x
    else:
        return x


def translate(x: np.ndarray, shift=0.2, p=0.5):
    if np.random.rand() < p:
        return x + np.random.uniform(-shift, shift, size=x.shape[-1])
    else:
        return x


def anisotropic_scale(x: np.ndarray, min_scale=0.8, max_scale=1.2, p=0.5):
    # x: (n, 3)
    if np.random.rand() < p:
        scale = np.random.uniform(min_scale, max_scale, size=x.shape[-1])
        return x * scale
    else:
        return x


class ShapeNetPart(Dataset):
    cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
              'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
              'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
    cls2parts = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
                 [16, 17, 18], [19, 20, 21], [22, 23], [24, 25, 26, 27], [28, 29],
                 [30, 31, 32, 33, 34, 35], [36, 37], [38, 39, 40], [41, 42, 43], [44, 45, 46], [47, 48, 49]]

    def __init__(
            self,
            n_points=2048,
            partition='trainval',
            class_choice=None,
            *,
            # rotate_func=partial(rotate, axis='y', angle=15, p=1.),
            anisotropic_scale_func=partial(anisotropic_scale, min_scale=0.66, max_scale=1.5, p=1.),
            jitter_func=partial(jitter, sigma=0.01, clip=0.05, p=1.),
            translate_func=partial(translate, shift=0.2, p=1.),
    ):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.n_points = n_points
        self.partition = partition
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

        # self.rotate = rotate_func
        self.jitter = jitter_func
        self.translate = translate_func
        self.anisotropic_scale = anisotropic_scale_func

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.n_points]
        label = self.label[item]
        seg = self.seg[item][:self.n_points]
        if self.partition == 'trainval':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            pointcloud = self.anisotropic_scale(pointcloud)
            pointcloud = self.jitter(pointcloud)
            pointcloud = self.translate(pointcloud)
            seg = seg[indices]
        return pointcloud.astype('f4'), label, seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    ShapeNetPart()[0]