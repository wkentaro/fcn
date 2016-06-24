#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys

import fcn
import scipy.ndimage as ndi
import skimage.io
import skimage.transform

import apc2016

sys.path.insert(0, '../apc2015')
import forward  # NOQA


this_dir = osp.dirname(osp.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only')
    parser.add_argument('-c', '--chainermodel')
    args = parser.parse_args()

    gpu = args.gpu
    chainermodel = args.chainermodel

    dataset = apc2016.APC2016Dataset()
    target_names = dataset.target_names

    forwarding = forward.Forwarding(gpu, target_names, chainermodel)

    stat = {
        'acc': 0,
        'acc_cls': 0,
        'mean_iu': 0,
        'fwavcc': 0,
    }
    n_data = len(dataset.val)
    for datum in dataset.val:
        img, label_true = dataset.load_datum(datum, train=False)
        img, label_pred, _ = forwarding.forward_img_file(datum['img_file'])

        bin_mask_path = datum['bin_mask_file']
        bin_mask = ndi.imread(bin_mask_path, mode='L')
        bin_mask = skimage.transform.resize(bin_mask, img.shape[:2],
                                            preserve_range=True)
        img[bin_mask == 0] = 0
        label_pred[bin_mask == 0] = 0

        acc, acc_cls, mean_iu, fwavcc = fcn.util.label_accuracy_score(
            label_true, label_pred, len(target_names))
        stat['acc'] += acc
        stat['acc_cls'] += acc_cls
        stat['mean_iu'] += mean_iu
        stat['fwavcc'] += fwavcc

        out_img = forwarding.visualize_label(img, label_pred)
        out_dir = osp.join(this_dir, 'forward_out')
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        out_img_path = osp.join(out_dir, datum['id'] + '.png')
        skimage.io.imsave(out_img_path, out_img)
        print('saved {}'.format(out_img_path))
    stat['acc'] /= n_data
    stat['acc_cls'] /= n_data
    stat['mean_iu'] /= n_data
    stat['fwavcc'] /= n_data
    print('''
acc: {acc}
acc_cls: {acc_cls}
mean_iu: {mean_iu}
fwavcc: {fwavcc}'''.format(**stat))


if __name__ == '__main__':
    main()
