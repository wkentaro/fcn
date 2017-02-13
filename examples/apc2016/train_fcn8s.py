#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda

import fcn

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fcn16s', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dataset', default='v2', choices=['v1', 'v2'])
    args = parser.parse_args()

    fcn16s_path = args.fcn16s
    gpu = args.gpu
    out = args.out
    if args.dataset == 'v1':
        dataset_class = datasets.APC2016DatasetV1
    else:
        dataset_class = datasets.APC2016DatasetV2

    if not osp.exists(out):
        os.makedirs(out)

    # 1. dataset

    dataset_train = dataset_class('train')
    dataset_val = dataset_class('val')

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(dataset_val, batch_size=1,
                                                repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.label_names)

    fcn16s = fcn.models.FCN16s(n_class=n_class)
    chainer.serializers.load_hdf5(fcn16s_path, fcn16s)

    model = fcn.models.FCN8s(n_class=n_class)
    model.train = True
    fcn.utils.copy_chainermodel(fcn16s, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)

    # training loop

    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_val=iter_val,
        out=out,
    )
    trainer.train(
        max_iter=150000,
        interval_eval=5000,
    )


if __name__ == '__main__':
    main()
