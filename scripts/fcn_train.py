#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import chainer.optimizers as O
import chainer.serializers as S

import fcn
from fcn.models import FCN32s
from fcn.models import VGG16
from fcn import pascal


def get_vgg16_pretrained_model(self):
    vgg16_path = fcn.setup.download_vgg16_chainermodel()
    vgg16 = VGG16()
    print('Loading vgg16 model: {0}'.format(vgg16_path))
    S.load_hdf5(vgg16_path, vgg16)
    return vgg16


def main():
    gpu = 0

    # setup dataset
    dataset = pascal.SegmentationClassDataset()
    n_class = len(dataset.target_names)

    # setup model
    pretrained_model = get_vgg16_pretrained_model()
    model = FCN32s(n_class=n_class)
    fcn.util.copy_chainermodel(pretrained_model, model)
    if gpu != -1:
        model.to_gpu(gpu)

    # setup optimizer
    optimizer = O.MomentumSGD(lr=1e-10, momentum=0.99)
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
