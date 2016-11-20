#!/usr/bin/env python

import argparse

import fcn


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
        dataset_class=fcn.datasets.PascalVOC2012SegmentationDataset,
        gpu=gpu,
        max_iter=max_iter,
        out=out,
        resume=resume,
    )
    trainer.run()


if __name__ == '__main__':
    main()
