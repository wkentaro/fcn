import glob
import os
import os.path as osp
import re

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016DatasetBase


class APC2016rboDataset(APC2016DatasetBase):

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        self.dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/APC2016rbo')
        data_ids = self._get_ids()
        ids_train, ids_val = train_test_split(
            data_ids, test_size=0.25, random_state=1234)
        if data_type == 'train':
            self._ids = ids_train
        else:
            self._ids = ids_val

    def __len__(self):
        return len(self._ids)

    def _get_ids(self):
        ids = []
        for img_file in os.listdir(self.dataset_dir):
            if not re.match(r'^.*_[0-9]*_bin_[a-l].jpg$', img_file):
                continue
            data_id = osp.splitext(img_file)[0]
            ids.append(data_id)
        return ids

    def _load_from_id(self, data_id):
        img_file = osp.join(self.dataset_dir, data_id + '.jpg')
        img = scipy.misc.imread(img_file)
        # generate label from mask files
        lbl = np.zeros(img.shape[:2], dtype=np.int32)
        # shelf bin mask file
        shelf_bin_mask_file = osp.join(self.dataset_dir, data_id + '.pbm')
        shelf_bin_mask = scipy.misc.imread(shelf_bin_mask_file, mode='L')
        lbl[shelf_bin_mask < 127] = -1
        # object mask files
        mask_glob = osp.join(self.dataset_dir, data_id + '_*.pbm')
        for mask_file in glob.glob(mask_glob):
            mask_id = osp.splitext(osp.basename(mask_file))[0]
            mask = scipy.misc.imread(mask_file, mode='L')
            lbl_name = mask_id[len(data_id + '_'):]
            lbl_id = self.label_names.index(lbl_name)
            lbl[mask > 127] = lbl_id
        return img, lbl

    def get_example(self, i):
        data_id = self._ids[i]
        img, lbl = self._load_from_id(data_id)
        datum = self.img_to_datum(img)
        return datum, lbl


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import six
    dataset_train = APC2016rboDataset('train')
    dataset_val = APC2016rboDataset('val')
    print('train: %d, val: %d' % (len(dataset_train), len(dataset_val)))
    for i in six.moves.range(len(dataset_val)):
        viz = dataset_val.visualize_example(i)
        plt.imshow(viz)
        plt.show()
