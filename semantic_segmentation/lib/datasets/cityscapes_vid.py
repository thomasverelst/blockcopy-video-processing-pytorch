import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import cv2

def default_loader(path):
    return Image.open(path).convert('RGB')
    # return Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

class CityscapesVid(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts

    mean = (73.1584 / 255, 82.9090 / 255, 72.3924 / 255)
    std = (44.9149 / 255, 46.1529 / 255, 45.3192 / 255)
    
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    fine_classes = [6,7,11,12,13,14,15,16,17,18]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    train_id_to_id = np.array([ c.id for c in classes if c.train_id < 255  ])
    # print('tid', train_id_to_id)
    
    train_id_to_name = [[] for i in range(20)]
    for c in classes:
        train_id = c.train_id
        if train_id == 255:
            train_id = 19
        train_id_to_name[train_id].append(c.name)
    train_id_to_name = [', '.join(t) for t in train_id_to_name]
    # print('train id to name', train_id_to_name)
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', target_type='semantic', transform=None, clip_length=20, has_labels=True):
        # preload_dir = None
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.vid_dir = os.path.join(self.root, 'leftImg8bit_sequence', split)
        self.extension = '.png'

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        assert clip_length > 0 and clip_length <= 20, 'Clip length must be between 1 and 20 frames'
        self.clip_length = clip_length
        self.interval = 1
        self.has_labels = has_labels

        self.split = split
        self.images = []
        self.cities = []
        self.relative_dirs = []
        self.file_names = []
        self.targets = []

        self.loader = default_loader
        

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir) or not os.path.isdir(self.vid_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are available:'
                               f'\n images dir: {self.images_dir}, found:  {os.path.isdir(self.images_dir)}'
                               f'\n images dir: {self.vid_dir}, found:  {os.path.isdir(self.vid_dir)}'
                               f'\n targets dir: {self.targets_dir}, found:  {os.path.isdir(self.targets_dir)}'
                               )

        for city in os.listdir(self.images_dir): 
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.relative_dirs.append(os.path.join(city, file_name))
                self.images.append(os.path.join(img_dir, file_name))
                self.file_names.append(file_name)
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))
        print(f'Cityscapes {split} split with {len(self.images)} videos')

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def encode_target_test(cls, target):
        return cls.train_id_to_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        # load main image
        image = self.loader(self.images[index])
        if self.has_labels:
            target = Image.open(self.targets[index])
        else:
            target=  None
        if self.transform:
            image, target = self.transform(image, target)
        if target is not None:
            target = self.encode_target(target)
        
        # decompose filename in order to get clip file paths
        fn = self.relative_dirs[index]
        fn = fn.replace('_leftImg8bit.png', '')
        fn_parts = fn.split('_')

        fn_prefix = '_'.join(fn_parts[:-1])
        fn_frame_id = int(fn_parts[-1])
        fn_suffix = '_leftImg8bit'+self.extension

        # build video tensor
        vid = [image,]

        # add video frames
        for i in range(1, self.clip_length):
            this_frame_id = fn_frame_id - i*self.interval
            this_fn = fn_prefix+'_'+str(this_frame_id).zfill(6)+fn_suffix
            path = os.path.join(self.vid_dir, this_fn)
            image = self.loader(path)
            if self.transform:
                image, _ = self.transform(image, None)
            vid.append(image)
        
        vid = vid[::-1] # reverse list of frames
        meta = {
            'relpath': self.relative_dirs[index]
        }
        if target is None:
            target = 0
        return vid, target, meta

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)