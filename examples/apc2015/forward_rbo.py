#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp

import fcn
import scipy.ndimage as ndi
import skimage.transform
import skimage.io

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

    for img_path in img_paths:
        img_path = osp.join(this_dir, 'dataset', img_path)
        img, label = forwarding.forward_img_file(img_path)

        mask_path = osp.splitext(img_path)[0] + '.pbm'
        mask = ndi.imread(mask_path, mode='L')
        mask = skimage.transform.resize(mask, img.shape[:2],
                                        preserve_range=True)
        img[mask == 0] = 0
        label[mask == 0] = 0

        out_img = forwarding.visualize_label(img, label)
        out_dir = osp.join(this_dir, 'forward_out')
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        out_img_path = osp.join(out_dir, osp.basename(img_path))
        skimage.io.imsave(out_img_path, out_img)
        print('saved {}'.format(out_img_path))


if __name__ == '__main__':
    main()
