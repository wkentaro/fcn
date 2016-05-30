#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
import glob
import os.path as osp
import re
import tempfile

import numpy as np
import plyvel
from scipy.misc import imread
import skimage.morphology
import skimage.transform
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch
import tqdm

import fcn


this_dir = osp.dirname(osp.realpath(__file__))


class APC2015(Bunch):

    def __init__(self, db_path):
        self.n_transforms = 6
        self.db = plyvel.DB(db_path, create_if_missing=True)

        self.target_names = [
            'background',
            'champion_copper_plus_spark_plug',
            'cheezit_big_original',
            'crayola_64_ct',
            'dr_browns_bottle_brush',
            'elmers_washable_no_run_school_glue',
            'expo_dry_erase_board_eraser',
            'feline_greenies_dental_treats',
            'first_years_take_and_toss_straw_cup',
            'genuine_joe_plastic_stir_sticks',
            'highland_6539_self_stick_notes',
            'kong_air_dog_squeakair_tennis_ball',
            'kong_duck_dog_toy',
            'kong_sitting_frog_dog_toy',
            'kyjen_squeakin_eggs_plush_puppies',
            'laugh_out_loud_joke_book',
            'mark_twain_huckleberry_finn',
            'mead_index_cards',
            'mommys_helper_outlet_plugs',
            'munchkin_white_hot_duck_bath_toy',
            'oreo_mega_stuf',
            'paper_mate_12_count_mirado_black_warrior',
            'rolodex_jumbo_pencil_cup',
            'safety_works_safety_glasses',
            'sharpie_accent_tank_style_highlighters',
            'stanley_66_052',
        ]
        self.n_class = len(self.target_names)
        self.mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

        self.ids = []
        self.img_files = []
        self.mask_files = []
        self.rois = []

        self.datasets = defaultdict(list)
        self._load_berkeley()
        self._load_rbo()
        for name, ids in self.datasets.items():
            print('Loaded {0}: {1}'.format(name, len(ids)))
        n_ids = len(self.ids)
        assert n_ids == len(set(self.ids))
        assert n_ids == len(self.img_files)
        assert n_ids == len(self.mask_files)
        assert n_ids == len(self.rois)

        self.ids = np.array(self.ids)
        self.img_files = np.array(self.img_files)
        self.mask_files = np.array(self.mask_files)

        seed = np.random.RandomState(1234)
        self.train, self.val = train_test_split(
            self.ids, test_size=0.2, random_state=seed)

    def _load_berkeley(self):
        """Load APC2015berkeley dataset"""
        dataset_dir = osp.join(this_dir, 'dataset/APC2015berkeley')
        for label_value, label_name in enumerate(self.target_names):
            img_file_glob = osp.join(
                dataset_dir, label_name, '*.jpg')
            desc = 'berkeley:%s' % label_name
            for img_file in tqdm.tqdm(glob.glob(img_file_glob),
                                      ncols=80, desc=desc):
                if np.random.randint(100) != 0:
                    continue
                img_id = re.sub('.jpg$', '', osp.basename(img_file))
                mask_file = osp.join(dataset_dir, label_name, 'masks',
                                     img_id + '_mask.jpg')
                id_ = osp.join('berkeley', label_name, img_id)
                dataset_index = len(self.ids) - 1
                self.datasets['berkeley'].append(dataset_index)
                mask_files = [None] * self.n_class
                mask_files[label_value] = mask_file
                # compute roi
                img = imread(img_file, mode='RGB')
                y_min, x_min = img.shape[0] // 3, img.shape[1] // 3
                y_max, x_max = y_min + img.shape[0]//3, x_min + img.shape[1]//3
                roi = (y_min, x_min), (y_max, x_max)
                del img
                # set to data fields
                self.ids.append(id_)
                self.rois.append(roi)
                self.img_files.append(img_file)
                self.mask_files.append(mask_files)

    def _load_rbo(self):
        """Load APC2015rbo dataset"""
        dataset_dir = osp.join(this_dir, 'dataset/APC2015rbo/berlin_samples')
        img_glob = osp.join(dataset_dir, '*_bin_[A-L].jpg')
        desc = 'rbo'
        for img_file in tqdm.tqdm(glob.glob(img_glob), ncols=80, desc=desc):
            basename = osp.splitext(osp.basename(img_file))[0]
            # apply mask, crop and save
            bin_mask_file = re.sub('.jpg$', '.pbm', img_file)
            bin_mask = imread(bin_mask_file, mode='L')
            where = np.argwhere(bin_mask)
            roi = where.min(0), where.max(0) + 1
            id_ = osp.join('rbo', basename)
            dataset_index = len(self.ids) - 1
            self.datasets['rbo'].append(dataset_index)
            mask_glob = re.sub('.jpg$', '_*.pbm', img_file)
            mask_files = [None] * self.n_class
            for mask_file in glob.glob(mask_glob):
                mask_basename = osp.splitext(osp.basename(mask_file))[0]
                label_name = re.sub(basename + '_', '', mask_basename)
                if label_name == 'shelf':
                    continue
                mask_files[self.target_names.index(label_name)] = mask_file
            self.ids.append(id_)
            self.rois.append(roi)
            self.img_files.append(img_file)
            self.mask_files.append(mask_files)

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

    def _load_datum(self, index, type):
        """Get inputs with global index (global means self.ids[index] works)"""
        max_size = 500 * 500
        roi = self.rois[index]
        img = imread(self.img_files[index], mode='RGB')
        if roi is not None:
            (y_min, x_min), (y_max, x_max) = roi
            img = img[y_min:y_max, x_min:x_max]
        img, _ = fcn.util.resize_img_with_max_size(img, max_size=max_size)
        label = np.zeros(img.shape[:2], dtype=np.int32)  # bg_label is 0
        height, width = img.shape[:2]
        translation = (int(0.1 * np.random.random() * height),
                       int(0.1 * np.random.random() * width))
        tform = skimage.transform.SimilarityTransform(translation=translation)
        img = skimage.transform.warp(
            img, tform, mode='constant', preserve_range=True).astype(np.uint8)
        for label_value, mask_file in enumerate(self.mask_files[index]):
            if mask_file is None:
                continue
            mask = imread(mask_file, mode='L')
            if roi is not None:
                (y_min, x_min), (y_max, x_max) = roi
                mask = mask[y_min:y_max, x_min:x_max]
            mask, _ = fcn.util.resize_img_with_max_size(mask, max_size)
            mask = skimage.transform.warp(
                mask, tform, mode='constant',
                preserve_range=True).astype(np.uint8)
            label[mask == 255] = label_value
        return img, label

    def next_batch(self, batch_size, type, indices=None):
        assert type in ('train', 'val')
        ids = getattr(self, type)
        n_data = len(ids)
        if indices is None:
            indices = np.random.randint(0, n_data, batch_size)
        batch = []
        for id_ in ids[indices]:
            index = np.where(self.ids == id_)[0][0]
            datum = self.db.get(str(id_))
            if datum is not None:
                # use cached data
                datum = pickle.loads(datum)
            else:
                datum = self._load_datum(index, type=type)
                if datum is None:
                    continue
                # save to db
                self.db.put(str(id_), pickle.dumps(datum))
            batch.append(datum)
        return batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    dataset = APC2015(tempfile.mktemp())
    batch = dataset.next_batch(batch_size=1, type='train')
    img, label = batch[0]
    print(np.unique(label))
    print(np.array(dataset.target_names)[np.unique(label)])
    label_viz = label2rgb(label, img, bg_label=0)
    plt.imshow(label_viz)
    plt.show()
