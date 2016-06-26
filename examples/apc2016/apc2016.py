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


class APC2016Dataset(object):

    target_names = np.array([
        'background',
        'barkely_hide_bones',
        'cherokee_easy_tee_shirt',
        'clorox_utility_brush',
        'cloud_b_plush_bear',
        'command_hooks',
        'cool_shot_glue_sticks',
        'crayola_24_ct',
        'creativity_chenille_stems',
        'dasani_water_bottle',
        'dove_beauty_bar',
        'dr_browns_bottle_brush',
        'easter_turtle_sippy_cup',
        'elmers_washable_no_run_school_glue',
        'expo_dry_erase_board_eraser',
        'fiskars_scissors_red',
        'fitness_gear_3lb_dumbbell',
        'folgers_classic_roast_coffee',
        'hanes_tube_socks',
        'i_am_a_bunny_book',
        'jane_eyre_dvd',
        'kleenex_paper_towels',
        'kleenex_tissue_box',
        'kyjen_squeakin_eggs_plush_puppies',
        'laugh_out_loud_joke_book',
        'oral_b_toothbrush_green',
        'oral_b_toothbrush_red',
        'peva_shower_curtain_liner',
        'platinum_pets_dog_bowl',
        'rawlings_baseball',
        'rolodex_jumbo_pencil_cup',
        'safety_first_outlet_plugs',
        'scotch_bubble_mailer',
        'scotch_duct_tape',
        'soft_white_lightbulb',
        'staples_index_cards',
        'ticonderoga_12_pencils',
        'up_glucose_bottle',
        'womens_knit_gloves',
        'woods_extension_cord',
    ])
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self):
        dataset = self.scrape()
        self.train, self.val = train_test_split(
            dataset, test_size=0.2, random_state=np.random.RandomState(1234))
        db_path = osp.join(this_dir, 'leveldb')
        self.db = plyvel.DB(db_path, create_if_missing=True)

    def scrape(self):
        # scrape rbo
        dataset_dir = osp.realpath(osp.join(this_dir, 'dataset/APC2016rbo'))
        dataset = []
        for file_ in os.listdir(dataset_dir):
            match = re.match('.*_[0-9]*_bin_[a-l].jpg$', file_)
            if match is None:
                continue
            basename = osp.splitext(file_)[0]
            img_file = osp.join(dataset_dir, basename + '.jpg')
            # json_file = osp.join(dataset_dir, basename + '.json')
            bin_mask_file = osp.join(dataset_dir, basename + '.pbm')
            # pkl_file = osp.join(dataset_dir, basename + '.pkl')
            mask_files = [None] * len(self.target_names)
            for mask_file in glob.glob(osp.join(dataset_dir,
                                                basename + '_*.pbm')):
                mask_file_basename = osp.basename(mask_file)
                match = re.match(basename + '_(.*).pbm', mask_file_basename)
                label_name = match.groups()[0]
                label_id = np.where(self.target_names == label_name)[0][0]
                mask_files[label_id] = mask_file
            dataset.append({
                'annotate_type': 'MaskImageList',
                'id': basename,
                'img_file': img_file,
                'bin_mask_file': bin_mask_file,
                'mask_files': mask_files,
            })
        # scrape seg
        dataset_dir = osp.realpath(
            osp.join(this_dir, 'dataset/APC2016seg/annotated'))
        for dir_ in os.listdir(dataset_dir):
            img_file = osp.join(dataset_dir, dir_, 'image.png')
            label_file = osp.join(dataset_dir, dir_, 'label.png')
            dataset.append({
                'annotate_type': 'LabelImage',
                'id': dir_,
                'img_file': img_file,
                'label_file': label_file,
            })
        return dataset

    def view_dataset(self):
        for datum in self.val:
            rgb, label = self.load_datum(datum, train=False)
            label_viz = label2rgb(label, rgb, bg_label=-1)
            label_viz[label == 0] = 0
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
        # # translate
        # height, width = rgb.shape[:2]
        # translation = (int(0.1 * np.random.random() * height),
        #                int(0.1 * np.random.random() * width))
        # tform = skimage.transform.SimilarityTransform(translation=translation)
        # rgb = skimage.transform.warp(
        #     rgb, tform, mode='constant', preserve_range=True)
        # rgb = rgb.astype(np.uint8)
        rgb = ndi.imread(datum['img_file'], mode='RGB')
        if datum['annotate_type'] == 'MaskImageList':
            max_size = 500 * 500
            rgb, _ = fcn.util.resize_img_with_max_size(rgb, max_size=max_size)
            # bin_mask
            bin_mask = ndi.imread(datum['bin_mask_file'], mode='L')
            bin_mask, _ = fcn.util.resize_img_with_max_size(
                bin_mask, max_size=max_size)
            # bin_mask = skimage.transform.warp(
            #     bin_mask, tform, mode='constant', preserve_range=True)
            # bin_mask = bin_mask.astype(np.uint8)
            # generate label
            label = np.zeros(rgb.shape[:2], dtype=np.int32)
            for label_value, mask_file in enumerate(datum['mask_files']):
                if mask_file is None:
                    continue
                mask = ndi.imread(mask_file, mode='L')
                mask, _ = fcn.util.resize_img_with_max_size(
                    mask, max_size=max_size)
                # mask = skimage.transform.warp(
                #     mask, tform, mode='constant', preserve_range=True)
                # mask = mask.astype(np.uint8)
                label[mask != 0] = label_value
            label[bin_mask == 0] = -1
            target_names = np.array(self.target_names.tolist() + ['unlabeled'])
        elif datum['annotate_type'] == 'LabelImage':
            label = ndi.imread(datum['label_file'], mode='L')
            label = label.astype(np.int32)
            label[label == 255] = -1
            target_names = np.array(self.target_names.tolist() + ['unlabeled'])
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
    dataset = APC2016Dataset()
    dataset.scrape()
    dataset.view_dataset()
