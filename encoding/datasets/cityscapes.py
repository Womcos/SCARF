
import os
import sys
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter

import torch
import re
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm
from .base import BaseDataset

class CitySegmentation(BaseDataset):
    NUM_CLASS = 19
    def __init__(self, root=os.path.expanduser('~/.encoding/data/citys/'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CitySegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        assert os.path.exists(root), "Please download the dataset!!"
        self.images, self.masks = get_city_pairs(self.root, self.split)
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def get_city_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        data = open(split_f, 'r')
        for line in data.readlines():
            img_name, mask_name = line.strip('\n').split(' ')
            img_path = os.path.join(folder, img_name)
            mask_path = os.path.join(folder, mask_name)
            if os.path.isfile(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print('cannot find the mask:', mask_path)
        return img_paths, mask_paths
    if split == 'train':
        split_f = os.path.join(folder, 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'trainval':
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        raise Exception('wrong split.')
    return img_paths, mask_paths
