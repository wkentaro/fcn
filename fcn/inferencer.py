from __future__ import division

import os.path as osp

import chainer
import numpy as np
import scipy.misc

from fcn import utils


class Inferencer(object):

    def __init__(self, dataset, model, gpu):
        self.dataset = dataset
        self.gpu = gpu
        self.model = model

        self.label_names = self.dataset.label_names

        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def infer(self, x):
        self.model.train = False
        self.model(x)
        score = chainer.cuda.to_cpu(self.model.score.data)[0]
        label = np.argmax(score, axis=0)
        return label

    def infer_image_file(self, img_file):
        print('{0}:'.format(osp.realpath(img_file)))
        # setup input
        img = scipy.misc.imread(img_file, mode='RGB')
        img, resizing_scale = utils.resize_img_with_max_size(img)
        print(' - resizing_scale: {0}'.format(resizing_scale))
        datum = self.dataset.img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data, device=self.gpu)
        x = chainer.Variable(x_data, volatile=False)
        label = self.infer(x)
        return img, label

    def visualize_label(self, img, label):
        # visualize result
        unique_labels, label_counts = np.unique(label, return_counts=True)
        print('- labels:')
        label_titles = {}
        for label_value, label_count in zip(unique_labels, label_counts):
            label_region = label_count / label.size
            if label_region < 0.001:
                continue
            title = '{0}:{1} = {2:.1%}'.format(
                label_value, self.label_names[label_value], label_region)
            label_titles[label_value] = title
            print('  - {0}'.format(title))
        labelviz = utils.draw_label(
            label, img, n_class=len(self.label_names),
            label_titles=label_titles)
        # save result
        return utils.get_tile_image([img, labelviz])
