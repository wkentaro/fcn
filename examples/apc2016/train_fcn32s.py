#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer.training import extensions

import fcn

import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-o', '--out', default='logs/latest')
    parser.add_argument('--resume')
    args = parser.parse_args()

    gpu = args.gpu
    out = args.out
    resume = args.resume
    max_iter = 100000

    trainer = fcn.trainers.fcn32s.get_trainer(
        dataset_class=dataset.APC2016Dataset,
        gpu=gpu,
        max_iter=max_iter,
        out=out,
        resume=resume,
    )
    trainer.run()


if __name__ == '__main__':
    main()
