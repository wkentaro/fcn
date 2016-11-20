#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import glob
import os.path as osp

import chainer
import numpy as np
import scipy.misc
import skimage.color
from sklearn.model_selection import train_test_split
import fcn


class APC2015Dataset(chainer.dataset.DatasetMixin):

    label_names = [
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
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self, data_type):
        assert data_type in ('train', 'val')

        self.ids = []
        self.dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2015/APC2015rbo/berlin_samples')
        img_glob = osp.join(self.dataset_dir, '*_bin_[A-L].jpg')
        for img_file in glob.glob(img_glob):
            id_ = osp.splitext(osp.basename(img_file))[0]
            self.ids.append(id_)

        seed = np.random.RandomState(1234)
        ids_train, ids_val = train_test_split(
            self.ids, test_size=0.2, random_state=seed)
        self.ids = ids_train if data_type == 'train' else ids_val

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        data_id = self.ids[i]

        img_file = osp.join(self.dataset_dir, data_id + '.jpg')
        img = scipy.misc.imread(img_file)
        datum = self.img_to_datum(img)

        label = np.zeros(img.shape[:2], dtype=np.int32)

        shelf_bin_mask_file = osp.join(
            self.dataset_dir, data_id + '.pbm')
        shelf_bin_mask = scipy.misc.imread(shelf_bin_mask_file, mode='L')
        label[shelf_bin_mask < 127] = -1

        mask_glob = osp.join(self.dataset_dir, data_id + '_*.pbm')
        for mask_file in glob.glob(mask_glob):
            mask_id = osp.splitext(osp.basename(mask_file))[0]
            mask = scipy.misc.imread(mask_file, mode='L')
            label_name = mask_id[len(data_id + '_'):]

            # typo in filename
            if label_name == 'laugh_out_load_joke_book':
                label_name = 'laugh_out_loud_joke_book'

            if label_name != 'shelf':
                label_value = self.label_names.index(label_name)
                label[mask > 127] = label_value

        return datum, label

    def visualize_example(self, i):
        n_class = len(self.label_names)
        cmap = fcn.util.labelcolormap(n_class)
        datum, label = self.get_example(i)
        img = self.datum_to_img(datum)
        ignore_label_mask = label == -1
        label[ignore_label_mask] = 0
        label_viz = skimage.color.label2rgb(
            label, img, colors=cmap, bg_label=0)
        label_viz = (label_viz * 255).astype(np.uint8)
        label_viz[ignore_label_mask] = (0, 0, 0)
        return label_viz

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = APC2015Dataset('val')
    for i in xrange(len(dataset)):
        labelviz = dataset.visualize_example(i)
        plt.imshow(labelviz)
        plt.show()
