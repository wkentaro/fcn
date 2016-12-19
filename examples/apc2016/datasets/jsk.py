import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016DatasetBase


class APC2016jskDataset(APC2016DatasetBase):

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        self.dataset_dir = chainer.dataset.get_dataset_directory(
            'apc2016/annotated')
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
        for data_id in os.listdir(self.dataset_dir):
            ids.append(data_id)
        return ids

    def _load_from_id(self, data_id):
        img_file = osp.join(self.dataset_dir, data_id, 'image.png')
        img = scipy.misc.imread(img_file)
        lbl_file = osp.join(self.dataset_dir, data_id, 'label.png')
        lbl = scipy.misc.imread(lbl_file, mode='L')
        lbl = lbl.astype(np.int32)
        lbl[lbl == 255] = -1
        return img, lbl

    def get_example(self, i):
        data_id = self._ids[i]
        img, lbl = self._load_from_id(data_id)
        datum = self.img_to_datum(img)
        return datum, lbl


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset_train = APC2016jskDataset('train')
    dataset_val = APC2016jskDataset('val')
    print('train: %d, val: %d' % (len(dataset_train), len(dataset_val)))
    for i in xrange(len(dataset_val)):
        viz = dataset_val.visualize_example(i)
        plt.imshow(viz)
        plt.show()
