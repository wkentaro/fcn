#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.training import extensions
import chainercv
import fcn


here = osp.dirname(osp.abspath(__file__))


def get_data():
    dataset_train = fcn.datasets.SBDClassSeg(split='train')

    class_names = dataset_train.class_names

    dataset_train = chainer.datasets.TransformDataset(
        dataset_train, fcn.datasets.transform_lsvrc2012_vgg16)
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)

    dataset_valid = fcn.datasets.VOC2011ClassSeg(split='seg11valid')
    iter_valid_raw = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)
    dataset_valid = chainer.datasets.TransformDataset(
        dataset_valid, fcn.datasets.transform_lsvrc2012_vgg16)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    return class_names, iter_train, iter_valid, iter_valid_raw


def get_trainer(optimizer, iter_train, iter_valid, iter_valid_raw,
                class_names, args):
    model = optimizer.target

    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(
        updater, (args.max_iteration, 'iteration'), out=args.out)

    trainer.extend(fcn.extensions.ParamsReport(args.__dict__))

    trainer.extend(extensions.ProgressBar(update_interval=5))

    trainer.extend(extensions.LogReport(
        trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'main/loss', 'validation/main/miou']))

    def pred_func(x):
        model(x)
        return model.score

    trainer.extend(
        fcn.extensions.SemanticSegmentationVisReport(
            pred_func, iter_valid_raw,
            transform=fcn.datasets.transform_lsvrc2012_vgg16,
            class_names=class_names, device=args.gpu, shape=(4, 2)),
        trigger=(args.interval_eval, 'iteration'))

    trainer.extend(
        chainercv.extensions.SemanticSegmentationEvaluator(
            iter_valid, model, label_names=class_names),
        trigger=(args.interval_eval, 'iteration'))

    trainer.extend(extensions.snapshot_object(
        target=model, filename='model_best.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/miou',
            trigger=(args.interval_eval, 'iteration')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss'], x_key='iteration',
        file_name='loss.png', trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['validation/main/miou'], x_key='iteration',
        file_name='miou.png', trigger=(args.interval_print, 'iteration')))

    return trainer


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    args = parser.parse_args()

    args.model = 'FCN32s'
    args.lr = 1e-10
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
    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg.pretrained_model, vgg)
    model = fcn.models.FCN32s(n_class=n_class)
    model.init_from_vgg16(vgg)

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
    model.upscore.disable_update()

    # trainer
    trainer = get_trainer(optimizer, iter_train, iter_valid, iter_valid_raw,
                          class_names, args)
    trainer.run()


if __name__ == '__main__':
    main()
