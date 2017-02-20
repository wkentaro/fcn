#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda

import fcn
from fcn.datasets import PascalVOC2012SegmentationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out

    if not osp.exists(out):
        os.makedirs(out)

    # 1. dataset

    dataset_train = PascalVOC2012SegmentationDataset('train')
    dataset_val = PascalVOC2012SegmentationDataset('val')

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(dataset_val, batch_size=1,
                                                repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.label_names)

    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    model = fcn.models.FCN32s(n_class=n_class)
    model.train = True
    fcn.utils.copy_chainermodel(vgg, model)
    for link_name in ['fc6', 'fc7']:
        W1, b1 = getattr(vgg, link_name).params()
        W2, b2 = getattr(vgg, link_name).params()
        W2.data = W1.data.reshape(W2.shape)
        b2.data = b1.data

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
