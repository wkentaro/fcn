#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import chainer
import chainer.optimizers as O
import chainer.serializers as S

import apc2016
import fcn
from fcn.models import FCN32s
from fcn.models import VGG16


def get_vgg16_pretrained_model(n_class):
    vgg16_path = fcn.setup.download_vgg16_chainermodel()
    vgg16 = VGG16()
    print('Loading vgg16 model: {0}'.format(vgg16_path))
    S.load_hdf5(vgg16_path, vgg16)
    return vgg16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    gpu = args.gpu

    dataset = apc2016.APC2016Dataset()
    n_class = len(dataset.target_names)

    # setup model
    pretrained_model = get_vgg16_pretrained_model(n_class)
    model = FCN32s(n_class=n_class)
    fcn.util.copy_chainermodel(pretrained_model, model)
    if gpu != -1:
        model.to_gpu(gpu)

    # setup optimizer
    # optimizer = O.MomentumSGD(lr=1e-8, momentum=0.99)
    optimizer = O.Adam(alpha=1e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))

    # train
    trainer = fcn.Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        weight_decay=0.0005,
        test_interval=1000,
        max_iter=20000,
        snapshot=1000,
        gpu=gpu,
    )
    trainer.train()


if __name__ == '__main__':
    main()
