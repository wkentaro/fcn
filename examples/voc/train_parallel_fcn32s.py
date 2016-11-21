#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer.training import extensions

import fcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=[0], nargs='*')
    parser.add_argument('-o', '--out-dir', default='logs/latest')
    parser.add_argument('--resume')
    args = parser.parse_args()

    gpus = args.gpus
    out_dir = args.out_dir
    resume = args.resume
    max_iter = 100000

    if not resume and osp.exists(out_dir):
        raise RuntimeError('Result dir already exists: {}'
                           .format(osp.abspath(out_dir)))

    # 1. dataset
    dataset_train = fcn.datasets.PascalVOC2012SegmentationDataset('train')
    dataset_val = fcn.datasets.PascalVOC2012SegmentationDataset('val')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=len(gpus), shared_mem=10000000)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=len(gpus), repeat=False, shuffle=False)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    model = fcn.models.FCN32s()
    model.train = True
    fcn.util.copy_chainermodel(vgg, model)

    chainer.cuda.get_device(gpus[0]).use()

    # 3. optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=1e-10, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # 4. trainer
    devices = {'main': gpus[0]}
    for gpu in gpus[1:]:
        devices['gpu{}'.format(gpu)] = gpu
    updater = chainer.training.ParallelUpdater(
        iter_train,
        optimizer,
        devices=devices,
    )
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out_dir)

    trainer.extend(
        fcn.training.extensions.TestModeEvaluator(
            iter_val, model, device=gpus[0]),
        trigger=(1000, 'iteration'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz',
        trigger=(1000, 'iteration')))
    trainer.extend(extensions.LogReport(
        trigger=(10, 'iteration'), log_name='log.json'))
    trainer.extend(extensions.PrintReport([
        'iteration',
        'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'main/iu', 'validation/main/iu',
        'elapsed_time',
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
