#!/usr/bin/env python

import argparse
import os.path as osp
import re

import chainer
from chainer import cuda
import fcn
import numpy as np
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()

    dataset = fcn.datasets.VOC2011ClassSeg('seg11valid')
    n_class = len(dataset.class_names)

    match = re.match('^fcn(32|16|8)s.*$',
                     osp.basename(args.model_file).lower())
    if match is None:
        print('Unsupported model filename: %s' % args.model_file)
        quit(1)
    model_name = 'FCN%ss' % match.groups()[0]
    model_class = getattr(fcn.models, model_name)
    model = model_class(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    lbl_preds, lbl_trues = [], []
    for i in tqdm.trange(len(dataset)):
        datum, lbl_true = dataset.get_example(i)
        x_data = np.expand_dims(datum, axis=0)
        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_data)

        with chainer.no_backprop_mode():
            x = chainer.Variable(x_data)
            with chainer.using_config('train', False):
                model(x)
                lbl_pred = chainer.functions.argmax(model.score, axis=1)[0]
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)

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
