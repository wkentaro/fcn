#!/usr/bin/env python

from __future__ import division

import argparse
import os.path as osp

import chainer
import scipy.misc

import fcn

from dataset import APC2016Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-c', '--chainermodel', required=True)
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    args = parser.parse_args()

    img_files = args.img_files
    gpu = args.gpu
    chainermodel = args.chainermodel
    save_dir = chainer.dataset.get_dataset_directory(
        'fcn/examples/apc2016/inference')

    dataset = APC2016Dataset('val')

    model = fcn.models.FCN8s(n_class=len(dataset.label_names))
    chainer.serializers.load_hdf5(chainermodel, model)

    infer = fcn.Inferencer(dataset, model, gpu)
    for img_file in img_files:
        img, label = infer.infer_image_file(img_file)
        out_img = infer.visualize_label(img, label)

        out_file = osp.join(save_dir, osp.basename(img_file))
        scipy.misc.imsave(out_file, out_img)
        print('- out_file: {0}'.format(out_file))


if __name__ == '__main__':
    main()
