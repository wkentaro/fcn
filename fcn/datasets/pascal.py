import os.path as osp

import chainer
import numpy as np
import scipy.misc

from fcn.datasets.segmentation_dataset import SegmentationDatasetBase


class PascalVOC2012SegmentationDataset(SegmentationDatasetBase):

    label_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_type, square_size=216):
        assert data_type in ('train', 'val')
        self.data_type = data_type
        self.square_size = square_size
        # get ids for the data_type
        dataset_dir = chainer.dataset.get_dataset_directory(
            'pascal/VOCdevkit/VOC2012')
        imgsets_file = osp.join(
            dataset_dir,
            'ImageSets/Segmentation/{}.txt'.format(data_type))
        self.files = []
        for data_id in open(imgsets_file).readlines():
            data_id = data_id.strip()
            img_file = osp.join(
                dataset_dir, 'JPEGImages/{}.jpg'.format(data_id))
            label_rgb_file = osp.join(
                dataset_dir, 'SegmentationClass/{}.png'.format(data_id))
            self.files.append({
                'img': img_file,
                'label_rgb': label_rgb_file,
            })

    def __len__(self):
        return len(self.files)

    def _fill_to_square(self, img, fill=0):
        if max(img.shape[:2]) > self.square_size:
            raise ValueError('Input image is too large: {}'.format(img.shape))
        shape = (self.square_size, self.square_size)
        if img.ndim == 3:
            shape = (shape[0], shape[1], img.shape[2])
        h, w = img.shape[:2]
        img_exp = np.zeros(shape, dtype=img.dtype)
        img_exp.fill(fill)
        img_exp[:h, :w] = img
        return img_exp

    def get_example(self, i):
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        img = scipy.misc.imread(img_file, mode='RGB')
        if self.data_type == 'train':
            height, width = img.shape[:2]
            scale = 1. * self.square_size / max(height, width)
            img = scipy.misc.imresize(img, size=scale, interp='bilinear')
            img = self._fill_to_square(img)
        # load label
        label_rgb_file = data_file['label_rgb']
        label_rgb = scipy.misc.imread(label_rgb_file, mode='RGB')
        label = self.label_rgb_to_32sc1(label_rgb)
        if self.data_type == 'train':
            label = scipy.misc.imresize(label, size=scale, interp='nearest')
            label = label.astype(np.int32)
            label = self._fill_to_square(label, fill=-1)
        return self.img_to_datum(img), label
