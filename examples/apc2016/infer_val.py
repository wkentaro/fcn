#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import fcn
import matplotlib
import numpy as np
import pandas
import six
import skimage.color
import skimage.io

from datasets import APC2016DatasetV1


def softmax(input):
    e = np.exp(input)
    return e / np.sum(e, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', required=True)
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-c', '--chainermodel', required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    gpu = args.gpu
    chainermodel = args.chainermodel

    dataset = APC2016DatasetV1('val')
    n_class = len(dataset.label_names)

    model = fcn.models.FCN32s(n_class=n_class)
    chainer.serializers.load_npz(chainermodel, model)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    records = []
    cmap = fcn.utils.labelcolormap(n_class)
    for i in six.moves.range(len(dataset)):
        sub_dir = osp.join(out_dir, '%06d' % i)
        if osp.exists(sub_dir):
            continue
        os.makedirs(sub_dir)

        datum, lbl_true = dataset.get_example(i)
        x_data = np.expand_dims(datum, axis=0)
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data, device=gpu)
        x = chainer.Variable(x_data, volatile=True)

        model(x)

        score = cuda.to_cpu(model.score.data[0])
        lbl_pred = score.argmax(axis=0)

        acc, acc_cls, iu, fwavacc = \
            fcn.utils.label_accuracy_score(lbl_true, lbl_pred, n_class)
        records.append((acc, acc_cls, iu, fwavacc))

        img = dataset.datum_to_img(datum)
        bin_mask = lbl_true != -1
        lbl_pred[~bin_mask] = 0
        lbl_true[~bin_mask] = 0
        viz_pred = skimage.color.label2rgb(
            lbl_pred, bg_label=0, colors=cmap[1:], bg_color=cmap[0])
        viz_true = skimage.color.label2rgb(
            lbl_true, bg_label=0, colors=cmap[1:], bg_color=cmap[0])

        img = fcn.utils.apply_mask(img, bin_mask, crop=True)
        viz_pred = fcn.utils.apply_mask(viz_pred, bin_mask, crop=True)
        viz_true = fcn.utils.apply_mask(viz_true, bin_mask, crop=True)

        skimage.io.imsave(osp.join(sub_dir, 'image.jpg'), img)
        skimage.io.imsave(osp.join(sub_dir, 'lbl_pred.jpg'), viz_pred)
        skimage.io.imsave(osp.join(sub_dir, 'lbl_true.jpg'), viz_true)

        prob = softmax(score)
        for c in six.moves.range(prob.shape[0]):
            cls_prob = prob[c]
            viz_prob = matplotlib.cm.jet(cls_prob)
            viz_prob = fcn.utils.apply_mask(viz_prob, bin_mask, crop=True)
            skimage.io.imsave(osp.join(sub_dir, 'prob_cls_%02d.jpg' % c),
                              viz_prob)
        print('saved to: %s' % sub_dir)

    columns = ['accuracy', 'accuracy_cls', 'iu', 'fwavacc']
    df = pandas.DataFrame(data=records, columns=columns)
    df.to_csv(osp.join(out_dir, 'log.csv'))


if __name__ == '__main__':
    main()
