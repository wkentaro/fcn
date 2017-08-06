#!/usr/bin/env python

import argparse
import os
import os.path as osp
import re

import chainer
import numpy as np
import skimage.io

import fcn


def infer(n_class):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('-m', '--model-file')
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    parser.add_argument('-o', '--out-dir', required=True)
    args = parser.parse_args()

    # model

    if args.model_file is None:
        args.model_file = fcn.models.FCN8s.download()

    match = re.match('^fcn(32|16|8)s.*$', osp.basename(args.model_file))
    if match is None:
        print('Unsupported model filename: %s' % args.model_file)
        quit(1)
    model_name = 'FCN%ss' % match.groups()[0]
    model_class = getattr(fcn.models, model_name)
    model = model_class(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # inference

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file in args.img_files:
        # input
        img = skimage.io.imread(file)
        lbl_dummy = np.zeros(img.shape[:2], dtype=np.int32)
        input, _ = fcn.datasets.VOC2012ClassSeg.transform(img, lbl_dummy)
        input = input[np.newaxis, :, :, :]
        if args.gpu >= 0:
            input = chainer.cuda.to_gpu(input)

        # forward
        with chainer.no_backprop_mode():
            input = chainer.Variable(input)
            with chainer.using_config('train', False):
                model(input)
                lbl_pred = chainer.functions.argmax(model.score, axis=1)[0]
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)

        # visualize
        label_titles = dict(
            enumerate(fcn.datasets.VOC2012ClassSeg.class_names))
        viz = fcn.utils.visualize_segmentation(
            lbl_pred=lbl_pred, img=img, n_class=n_class,
            label_names=label_titles)
        out_file = osp.join(args.out_dir, osp.basename(file))
        skimage.io.imsave(out_file, viz)
        print('==> wrote to: %s' % out_file)


if __name__ == '__main__':
    infer(n_class=21)
