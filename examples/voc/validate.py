#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer import cuda
import fcn
import numpy as np
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-c', '--chainermodel', required=True)
    args = parser.parse_args()

    dataset = fcn.datasets.VOC2011ClassSeg('seg11valid')
    n_class = len(dataset.class_names)

    model = fcn.models.FCN32s(n_class=n_class)
    if osp.splitext(args.chainermodel)[-1] == '.h5':
        chainer.serializers.load_hdf5(args.chainermodel, model)
    else:
        chainer.serializers.load_npz(args.chainermodel, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    lbl_preds, lbl_trues = [], []
    for i in tqdm.trange(len(dataset)):
        datum, lbl_true = dataset.get_example(i)
        x_data = np.expand_dims(datum, axis=0)
        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_data)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = chainer.Variable(x_data)
            model(x)

        score = cuda.to_cpu(model.score.data[0])
        lbl_pred = score.argmax(axis=0)

        lbl_preds.append(lbl_pred)
        lbl_trues.append(lbl_true)

    acc, acc_cls, mean_iu, fwavacc = \
        fcn.utils.label_accuracy_score(lbl_trues, lbl_preds, n_class)
    print('Accuracy: %.4f' % (100 * acc))
    print('AccClass: %.4f' % (100 * acc_cls))
    print('Mean IoU: %.4f' % (100 * mean_iu))
    print('Fwav Acc: %.4f' % (100 * fwavacc))


if __name__ == '__main__':
    main()
