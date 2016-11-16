#!/usr/bin/env python

import chainer
from chainer.training import extensions

import fcn


def main():
    gpu = 0
    resume = None  # filename

    # 1. dataset
    dataset_train = fcn.datasets.PascalVOC2012SegmentationDataset('train')
    dataset_val = fcn.datasets.PascalVOC2012SegmentationDataset('val')

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(dataset_val, batch_size=1)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    model = fcn.models.FCN32s()
    fcn.util.copy_chainermodel(vgg, model)

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=1e-10, momentum=0.99)
    optimizer.setup(model)

    # 4. trainer
    max_epoch = 10000
    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=gpu)
    trainer = chainer.training.Trainer(
        updater, (max_epoch, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(iter_val, model, device=gpu))
    trainer.extend(extensions.snapshot(), trigger=(max_epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if resume:
        chainer.serializers.load_hdf5(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
