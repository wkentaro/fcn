#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle
import glob
import os
import os.path as osp
import re

import fcn
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.transform
import skimage.io

import apc2015
import forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only')
    parser.add_argument('-c', '--chainermodel')
    args = parser.parse_args()

    gpu = args.gpu
    chainermodel = args.chainermodel

    forwarding = forward.Forwarding(gpu, chainermodel)

    this_dir = osp.dirname(osp.realpath(__file__))
    with open(osp.join(this_dir, 'dataset', 'rbo_val.txt'), 'r') as f:
        img_paths = [path.strip() for path in f.readlines()]

    target_names = apc2015.APC2015.target_names

    stat = {
        'acc': 0,
        'acc_cls': 0,
        'mean_iu': 0,
        'fwavcc': 0,
    }
    for img_path in img_paths:
        img_path = osp.join(this_dir, 'dataset', img_path)
        img, label_pred_all, pred = forwarding.forward_img_file(img_path)
        basename = osp.splitext(osp.basename(img_path))[0]

        mask_glob = re.sub('.jpg$', '_*.pbm', img_path)
        label_true = np.zeros_like(label_pred_all)
        for mask_file in glob.glob(mask_glob):
            mask_basename = osp.splitext(osp.basename(mask_file))[0]
            label_name = re.sub(basename + '_', '', mask_basename)
            if label_name == 'shelf':
                continue
            label_val = target_names.index(label_name)
            mask = ndi.imread(mask_file, mode='L')
            mask, _ = fcn.util.resize_img_with_max_size(mask)
            label_true[mask != 0] = label_val

        pkl_path = re.sub('.jpg$', '.pkl', img_path)
        pkl_data = pickle.load(open(pkl_path))
        candidate_labels = [target_names.index(obj)
                            for obj in ['background'] + pkl_data['objects']]

        label_pred = pred[candidate_labels].argmax(axis=0)
        for ind, label_val in enumerate(candidate_labels):
            label_pred[label_pred == ind] = label_val

        mask_path = osp.splitext(img_path)[0] + '.pbm'
        mask = ndi.imread(mask_path, mode='L')
        mask = skimage.transform.resize(mask, img.shape[:2],
                                        preserve_range=True)
        img[mask == 0] = 0
        label_pred[mask == 0] = 0

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
        out_img_path = osp.join(out_dir, osp.basename(img_path))
        skimage.io.imsave(out_img_path, out_img)
        print('saved {}'.format(out_img_path))
    stat['acc'] /= len(img_paths)
    stat['acc_cls'] /= len(img_paths)
    stat['mean_iu'] /= len(img_paths)
    stat['fwavcc'] /= len(img_paths)
    print('''
acc: {acc}
acc_cls: {acc_cls}
mean_iu: {mean_iu}
fwavcc: {fwavcc}'''.format(**stat))


if __name__ == '__main__':
    main()
