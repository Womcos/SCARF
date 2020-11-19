import os
import random
import scipy.io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from .base import BaseDataset

class VOCAugSegmentation2(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    BASE_DIR = 'VOCdevkit/VOC2012_seg_aug'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOCAugSegmentation2, self).__init__(root, split, mode, transform,
                                                 target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClassAug')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if self.mode == 'train':
            _split_f = os.path.join(_splits_dir, 'train_aug.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                _img = self.transform(_img)
            return _img, os.path.basename(self.images[index])
        _target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            _img, _target = self._sync_transform( _img, _target)
        elif self.mode == 'val':
            _img, _target = self._val_sync_transform( _img, _target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()
