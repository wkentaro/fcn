#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

import chainer.optimizers as O
import chainer.serializers as S

import apc2015
import fcn
from fcn.models import FCN32s
from fcn.models import VGG16


def download_vgg16_chainermodel():
    this_dir = osp.dirname(osp.abspath(__file__))
    vgg16_path = osp.join(this_dir, 'vgg16.chainermodel')
    fcn.util.download_data(
        pkg_name='fcn/example/apc2015',
        path=vgg16_path,
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vQ2tCN1hoYV84eHM',
        md5='4a48c2f39234e46937759f4cc43bb257',
    )
    return vgg16_path


def get_vgg16_pretrained_model(n_class):
    vgg16_path = download_vgg16_chainermodel()
    vgg16 = VGG16(n_class)
    print('Loading vgg16 model: {0}'.format(vgg16_path))
    S.load_hdf5(vgg16_path, vgg16)
    return vgg16


def main():
    gpu = 0

    this_dir = osp.dirname(osp.abspath(__file__))
    db_path = osp.join(this_dir, 'leveldb')

    dataset = apc2015.APC2015(db_path=db_path)
    n_class = len(dataset.target_names)

    # setup model
    pretrained_model = get_vgg16_pretrained_model(n_class)
    model = FCN32s(n_class=n_class)
    fcn.util.copy_chainermodel(pretrained_model, model)
    if gpu != -1:
        model.to_gpu(gpu)

    # setup optimizer
    optimizer = O.MomentumSGD(lr=1e-25, momentum=0.99)
    optimizer.setup(model)

    # train
    trainer = fcn.Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        weight_decay=0.0005,
        test_interval=1000,
        max_iter=100000,
        snapshot=4000,
        gpu=gpu,
    )
    trainer.train()


if __name__ == '__main__':
    main()
