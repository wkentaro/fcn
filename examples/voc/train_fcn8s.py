#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import subprocess

import chainer
from chainer import cuda

import fcn
from fcn import datasets


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU id')
    parser.add_argument(
        '--fcn16s-file',
        default=fcn.data.download_fcn16s_chainermodel(check_md5=False),
        help='Pretrained model file of FCN16s')
    args = parser.parse_args()

    gpu = args.gpu
    fcn16s_file = args.fcn16s_file

    # 0. config

    cmd = 'git log -n1 --format="%h"'
    vcs_version = subprocess.check_output(cmd, shell=True).strip()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = 'fcn8s_VCS-%s_TIME-%s' % (
        vcs_version,
        timestamp,
    )
    out = osp.join(here, 'logs', out)
    if not osp.exists(out):
        os.makedirs(out)
    with open(osp.join(out, 'config.yaml'), 'w') as f:
        f.write('fcn16s_file: %s\n' % fcn16s_file)

    # 1. dataset

    dataset_train = datasets.SBDClassSeg(split='train')
    dataset_valid = datasets.VOC2011ClassSeg(split='seg11valid')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=1, shared_mem=10**7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid, batch_size=1, shared_mem=10**7,
        repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.class_names)

    fcn16s = fcn.models.FCN16s()
    chainer.serializers.load_npz(fcn16s_file, fcn16s)

    model = fcn.models.FCN8s(n_class=n_class)
    model.init_from_fcn16s(fcn16s)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=1.0e-14, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    model.upscore2.disable_update()
    model.upscore8.disable_update()
    model.upscore_pool4.disable_update()

    # training loop

    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_valid=iter_valid,
        out=out,
        max_iter=100000,
    )
    trainer.train()


if __name__ == '__main__':
    main()
