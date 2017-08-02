import collections
import os.path as osp

import chainer
import numpy as np
import PIL.Image
import scipy.io

import fcn.data
import fcn.utils


DATASETS_DIR = osp.expanduser('~/data/datasets/VOC')


class VOCClassSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array([
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

    def __init__(self, year, split='train', transform=True):
        self.split = split
        self._transform = transform

        # VOC20XX is subset of VOC2012
        dataset_dir = osp.expanduser('~/data/datasets/VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def get_example(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    @classmethod
    def transform(cls, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= cls.mean_bgr
        img = img.transpose(2, 0, 1)
        return img, lbl

    @classmethod
    def untransform(cls, img, lbl):
        img = img.transpose(1, 2, 0)
        img += cls.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl

    @staticmethod
    def download():
        raise NotImplementedError


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, split='train', transform=True):
        super(VOC2011ClassSeg, self).__init__(
            year=2011, split=split, transform=transform)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'external/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        # VOC2011 is subset of VOC2012
        dataset_dir = osp.expanduser('~/data/datasets/VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})

    @staticmethod
    def download():
        VOC2012ClassSeg.download()


class VOC2012ClassSeg(VOCClassSegBase):

    def __init__(self, split='train', transform=True):
        super(VOC2012ClassSeg, self).__init__(
            year=2012, split=split, transform=transform)

    @staticmethod
    def download():
        url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
        path = osp.join(DATASETS_DIR, osp.basename(url))
        md5 = '6cd6e144f989b92b3379bac3b3de84fd'
        fcn.data.cached_download(url, path, md5)
        fcn.utils.extract_file(path, to_directory=DATASETS_DIR)


class SBDClassSeg(VOCClassSegBase):

    def __init__(self, split='train', transform=True):
        self.split = split
        self._transform = transform

        dataset_dir = osp.expanduser(
            '~/data/datasets/VOC/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def get_example(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    @staticmethod
    def download():
        # It must be renamed to benchmark.tar to be extracted
        url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA
        path = osp.join(DATASETS_DIR, 'benchmark.tar')
        md5 = '2b2af8a6cff7365684e002c08be823a6'
        fcn.data.cached_download(url, path, md5)
        fcn.utils.extract_file(path, to_directory=DATASETS_DIR)
