from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os.path as osp
import tempfile

import chainer
import numpy as np
import plyvel
import scipy.misc
import skimage.color

import fcn


class PascalVOC2012SegmentationDataset(chainer.dataset.DatasetMixin):

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

    def __init__(self, data_type):
        # set db
        self.db = plyvel.DB(tempfile.mktemp(), create_if_missing=True)
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

    def get_example(self, i):
        data_file = self.files[i]
        # load cache
        cache = self.db.get(str(i))
        if cache is not None:
            return pickle.loads(cache)
        # load image
        img_file = data_file['img']
        img = scipy.misc.imread(img_file, mode='RGB')
        datum = self.img_to_datum(img)
        # load label
        label_rgb_file = data_file['label_rgb']
        label_rgb = scipy.misc.imread(label_rgb_file, mode='RGB')
        label = self.label_rgb_to_32sc1(label_rgb)
        # store cache
        cache = (datum, label)
        self.db.put(str(i), pickle.dumps(cache))
        return datum, label

    def visualize_example(self, i):
        datum, label = self.get_example(i)
        img = self.datum_to_img(datum)
        cmap = fcn.util.labelcolormap(len(self.label_names))
        ignore_mask = [label == -1]
        label[ignore_mask] = 0
        labelviz = skimage.color.label2rgb(
            label, image=img, colors=cmap[1:], bg_label=0)
        labelviz[ignore_mask] = (0, 0, 0)
        labelviz = (labelviz * 255).astype(np.uint8)
        return labelviz

    def label_rgb_to_32sc1(self, label_rgb):
        assert label_rgb.dtype == np.uint8
        label = np.zeros(label_rgb.shape[:2], dtype=np.int32)
        label.fill(-1)
        cmap = fcn.util.labelcolormap(len(self.label_names))
        cmap = (cmap * 255).astype(np.uint8)
        for l, rgb in enumerate(cmap):
            mask = np.all(label_rgb == rgb, axis=-1)
            label[mask] = l
        return label

    def img_to_datum(self, img):
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= self.mean_bgr
        datum = datum.transpose((2, 0, 1))
        return datum

    def datum_to_img(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb
