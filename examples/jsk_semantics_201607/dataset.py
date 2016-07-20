from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import plyvel
import scipy.ndimage as ndi
from skimage.color import label2rgb
import skimage.transform
from sklearn.cross_validation import train_test_split

import fcn.util


this_dir = osp.dirname(osp.realpath(__file__))


class JskSemantics201607Dataset(object):

    target_names = np.array([
        'background',
        'room73b2-hitachi-fiesta-refrigerator',
        'room73b2-karimoku-table',
        'room73b2-hrp2-parts-drawer',
        'room73b2-door-left',
        'room73b2-door-right',
    ])
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self):
        dataset = self.scrape()
        self.train, self.val = train_test_split(
            dataset, test_size=0.2, random_state=np.random.RandomState(1234))
        print('Splitted dataset. total: {}, train: {}, val: {}'
              .format(len(dataset), len(self.train), len(self.val)))
        db_path = osp.join(this_dir, 'leveldb')
        self.db = plyvel.DB(db_path, create_if_missing=True)

    def scrape(self):
        # scrape rbo
        datasets_dir = osp.realpath(osp.join(this_dir, 'datasets'))
        dataset = []
        for dataset_dir in os.listdir(datasets_dir):
            dataset_dir = osp.join(datasets_dir, dataset_dir)
            if not osp.isdir(dataset_dir):
                continue
            for dir_ in os.listdir(dataset_dir):
                img_file = osp.join(dataset_dir, dir_, 'image.png')
                label_file = osp.join(dataset_dir, dir_, 'label.png')
                dataset.append({
                    'id': osp.join(dataset_dir, dir_),
                    'img_file': img_file,
                    'label_file': label_file,
                })
        return dataset

    def view_dataset(self):
        for datum in self.val:
            rgb, label = self.load_datum(datum, train=False)
            label_viz = label2rgb(label, rgb, bg_label=0)
            plt.imshow(label_viz)
            plt.show()

    def img_to_datum(self, rgb):
        rgb = rgb.astype(np.float32)
        blob = rgb[:, :, ::-1]  # RGB-> BGR
        blob -= self.mean_bgr
        blob = blob.transpose((2, 0, 1))
        return blob

    def datum_to_img(self, blob):
        bgr = blob.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb

    def load_datum(self, datum, train):
        rgb = ndi.imread(datum['img_file'], mode='RGB')
        rgb, _ = fcn.util.resize_img_with_max_size(rgb, max_size=500*500)
        # # translate
        # height, width = rgb.shape[:2]
        # translation = (int(0.1 * np.random.random() * height),
        #                int(0.1 * np.random.random() * width))
        # tform = skimage.transform.SimilarityTransform(translation=translation)
        # rgb = skimage.transform.warp(
        #     rgb, tform, mode='constant', preserve_range=True)
        # rgb = rgb.astype(np.uint8)
        label = ndi.imread(datum['label_file'], mode='L')
        label = label.astype(np.int32)
        label[label == 255] = -1
        # resize label carefully
        unique_labels = np.unique(label)
        label = skimage.transform.resize(
            label, rgb.shape[:2], order=0,
            preserve_range=True).astype(np.int32)
        np.testing.assert_array_equal(unique_labels, np.unique(label))
        return rgb, label

    def next_batch(self, batch_size, type, indices=None):
        assert type in ('train', 'val')
        if type == 'train':
            data = self.train
        else:
            data = self.val
        if indices is None:
            indices = np.random.randint(0, len(data), batch_size)
        batch = []
        for index in indices:
            datum = data[index]
            inputs = self.db.get(datum['id'])
            if inputs is not None:
                # use cached data
                inputs = pickle.loads(inputs)
            else:
                inputs = self.load_datum(datum, train=type == 'train')
                # save to db
                self.db.put(datum['id'], pickle.dumps(inputs))
            batch.append(inputs)
        return batch


if __name__ == '__main__':
    dataset = JskSemantics201607Dataset()
    dataset.scrape()
    dataset.view_dataset()
