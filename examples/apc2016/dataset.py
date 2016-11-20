import glob
import os
import os.path as osp
import re

import chainer
import numpy as np
import scipy.misc
import skimage.color
from sklearn.model_selection import train_test_split

import fcn


class APC2016Dataset(chainer.dataset.DatasetMixin):

    label_names = [
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
    ]
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if data_type == 'train' else iter_val

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        # APC2016rbo
        dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/APC2016rbo')
        for img_file in os.listdir(dataset_dir):
            if not re.match(r'^.*_[0-9]*_bin_[a-l].jpg$', img_file):
                continue
            data_id = osp.splitext(img_file)[0]
            ids.append(osp.join('rbo', data_id))
        # APC2016seg
        dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/APC2016JSKseg/annotated')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('JSKseg', data_id))
        return ids

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('rbo', 'JSKseg')

        if ann_id == 'rbo':
            dataset_dir = chainer.dataset.get_dataset_directory(
                'apc2016/APC2016rbo')

            img_file = osp.join(dataset_dir, data_id + '.jpg')
            img = scipy.misc.imread(img_file)
            datum = self.img_to_datum(img)

            label = np.zeros(img.shape[:2], dtype=np.int32)

            shelf_bin_mask_file = osp.join(
                dataset_dir, data_id + '.pbm')
            shelf_bin_mask = scipy.misc.imread(shelf_bin_mask_file, mode='L')
            label[shelf_bin_mask < 127] = -1

            mask_glob = osp.join(dataset_dir, data_id + '_*.pbm')
            for mask_file in glob.glob(mask_glob):
                mask_id = osp.splitext(osp.basename(mask_file))[0]
                mask = scipy.misc.imread(mask_file, mode='L')
                label_name = mask_id[len(data_id + '_'):]
                label_value = self.label_names.index(label_name)
                label[mask > 127] = label_value
        else:
            dataset_dir = chainer.dataset.get_dataset_directory(
                'apc2016/APC2016JSKseg/annotated')

            img_file = osp.join(dataset_dir, data_id, 'image.png')
            img = scipy.misc.imread(img_file)
            datum = self.img_to_datum(img)

            label_file = osp.join(dataset_dir, data_id, 'label.png')
            label = scipy.misc.imread(label_file, mode='L')
            label = label.astype(np.int32)
            label[label == 255] = -1
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
    dataset = APC2016Dataset('val')
    for i in xrange(len(dataset)):
        labelviz = dataset.visualize_example(i)
        plt.imshow(labelviz)
        plt.show()
