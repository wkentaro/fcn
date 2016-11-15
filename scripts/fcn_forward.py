#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os.path as osp

import chainer
import chainer.serializers as S
import numpy as np
import scipy.misc

import fcn


class Forwarding(object):

    def __init__(self, gpu, chainermodel=None):
        self.gpu = gpu

        self.target_names = fcn.pascal.SegmentationClassDataset.target_names
        self.n_class = len(self.target_names)

        if chainermodel is None:
            from fcn.models import FCN8s
            chainermodel = fcn.data.download_fcn8s_from_caffe_chainermodel()
            self.model_name = 'fcn8s'
            self.model = FCN8s(n_class=self.n_class)
        elif osp.basename(chainermodel).startswith('fcn8s'):
            from fcn.models import FCN8s
            self.model_name = 'fcn8s'
            self.model = FCN8s(n_class=self.n_class)
        elif osp.basename(chainermodel).startswith('fcn16s'):
            from fcn.models import FCN16s
            self.model_name = 'fcn16s'
            self.model = FCN16s(n_class=self.n_class)
        elif osp.basename(chainermodel).startswith('fcn32s'):
            from fcn.models import FCN32s
            self.model_name = 'fcn32s'
            self.model = FCN32s(n_class=self.n_class)
        else:
            raise ValueError(
                'Chainer model filename must start with fcn8s, '
                'fcn16s or fcn32s: {0}'.format(osp.basename(chainermodel)))

        S.load_hdf5(chainermodel, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def forward_img_file(self, img_file):
        print('{0}:'.format(osp.realpath(img_file)))
        # setup image
        img = scipy.misc.imread(img_file, mode='RGB')
        img, resizing_scale = fcn.util.resize_img_with_max_size(img)
        print(' - resizing_scale: {0}'.format(resizing_scale))
        # setup input datum
        datum = fcn.pascal.SegmentationClassDataset.img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        if self.gpu != -1:
            x_data = chainer.cuda.to_gpu(x_data, device=self.gpu)
        x = chainer.Variable(x_data, volatile=False)
        # forward
        self.model.train = False
        self.model(x)
        pred = self.model.score
        # generate computational_graph
        psfile = osp.join(
            chainer.dataset.get_dataset_directory('fcn'),
            '{0}_forward.ps'.format(self.model_name))
        if not osp.exists(psfile):
            fcn.util.draw_computational_graph([pred], output=psfile)
            print('- computational_graph: {0}'.format(psfile))
        pred_datum = chainer.cuda.to_cpu(pred.data)[0]
        label = np.argmax(pred_datum, axis=0)
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
                label_value, self.target_names[label_value], label_region)
            label_titles[label_value] = title
            print('  - {0}'.format(title))
        result_img = fcn.util.draw_label(
            label, img, n_class=self.n_class, label_titles=label_titles)
        # save result
        height, width = img.shape[:2]
        if height > width:
            vline = np.ones((height, 3, 3), dtype=np.uint8) * 255
            out_img = np.hstack((img, vline, result_img))
        else:
            hline = np.ones((3, width, 3), dtype=np.uint8) * 255
            out_img = np.vstack((img, hline, result_img))
        return out_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-c', '--chainermodel')
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    args = parser.parse_args()

    img_files = args.img_files
    gpu = args.gpu
    chainermodel = args.chainermodel
    save_dir = chainer.dataset.get_dataset_directory('fcn/fcn_forward')

    forwarding = Forwarding(gpu, chainermodel)
    for img_file in img_files:
        img, label = forwarding.forward_img_file(img_file)
        out_img = forwarding.visualize_label(img, label)

        out_file = osp.join(save_dir, osp.basename(img_file))
        scipy.misc.imsave(out_file, out_img)
        print('- out_file: {0}'.format(out_file))


if __name__ == '__main__':
    main()
