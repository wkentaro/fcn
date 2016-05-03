#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import os.path as osp

from chainer import cuda
import chainer.serializers as S
from chainer import Variable
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from skimage.color import label2rgb
from skimage.transform import rescale

import fcn
from fcn.models import FCN8s


def img_to_datum(img):
    datum = img.astype(np.float32)
    datum = datum[:, :, ::-1]  # RGB -> BGR
    datum -= np.array((104.00698793, 116.66876762, 122.67891434))
    datum = datum.transpose((2, 0, 1))
    return datum


class Forwarding(object):

    def __init__(self):
        data_dir = fcn.get_data_dir()
        chainermodel = osp.join(data_dir, 'fcn8s.chainermodel')

        self.target_names = fcn.pascal.get_target_names()
        self.model = FCN8s(n_class=len(self.target_names))
        S.load_hdf5(chainermodel, self.model)
        self.model.to_gpu()

    def forward_img_file(self, img_file):
        print('{0}:'.format(osp.realpath(img_file)))
        # setup image
        img = imread(img_file)
        scale =  (400 * 400) / (img.shape[0] * img.shape[1])
        if scale < 1:
            resizing_scale = np.sqrt(scale)
            print(' - resizing_scale: {0}'.format(resizing_scale))
            img = rescale(img, resizing_scale, preserve_range=True)
            img = img.astype(np.uint8)
        # setup input datum
        datum = img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        x_data = cuda.to_gpu(x_data)
        x = Variable(x_data, volatile=True)
        # forward
        self.model.train = False
        pred = self.model(x)
        pred_datum = cuda.to_cpu(pred.data)[0]
        label = np.argmax(pred_datum, axis=0)
        unique_labels = np.unique(label)
        print(' - unique_labels:', unique_labels)
        print(' - target_names:', self.target_names[unique_labels])
        # visualize
        cmap = fcn.util.labelcolormap(21)
        label_viz = label2rgb(label, img, colors=cmap[1:], bg_label=0)
        # plot image
        plt.axis('off')
        plt.imshow(label_viz)
        # plot legend
        plt_handlers = []
        plt_titles = []
        for l in np.unique(label):
            if l == np.where(self.target_names == 'background')[0][0]:
                continue  # skip background
            fc = cmap[l]
            p = plt.Rectangle((0, 0), 1, 1, fc=fc)
            plt_handlers.append(p)
            plt_titles.append(self.target_names[l])
        plt.legend(plt_handlers, plt_titles)
        plt.tight_layout()
        # save result
        data_dir = fcn.get_data_dir()
        basename = osp.splitext(osp.basename(img_file))[0]
        save_dir = osp.join(data_dir, 'forward_out')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        result_file = osp.join(save_dir, basename + '.jpg')
        plt.savefig(result_file)
        print(' - result_file: {0}'.format(result_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    args = parser.parse_args()

    img_files = args.img_files

    forwarding = Forwarding()
    for img_file in img_files:
        forwarding.forward_img_file(img_file)


if __name__ == '__main__':
    main()
