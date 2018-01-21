#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'

import chainer
import fcn

from train_fcn32s import get_data
from train_fcn32s import get_trainer


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument(
        '--fcn32s-file', default=fcn.models.FCN32s.pretrained_model,
        help='pretrained model file of FCN32s')
    args = parser.parse_args()

    args.model = 'FCN16s'
    args.lr = 1.0e-12
    args.momentum = 0.99
    args.weight_decay = 0.0005

    args.max_iteration = 100000
    args.interval_print = 20
    args.interval_eval = 4000

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # data
    class_names, iter_train, iter_valid, iter_valid_raw = get_data()
    n_class = len(class_names)

    # model
    fcn32s = fcn.models.FCN32s()
    chainer.serializers.load_npz(args.fcn32s_file, fcn32s)
    model = fcn.models.FCN16s(n_class=n_class)
    model.init_from_fcn32s(fcn32s)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(
        lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    model.upscore2.disable_update()
    model.upscore16.disable_update()

    # trainer
    trainer = get_trainer(optimizer, iter_train, iter_valid, iter_valid_raw,
                          class_names, args)
    trainer.run()


if __name__ == '__main__':
    main()
