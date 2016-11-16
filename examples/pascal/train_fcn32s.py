#!/usr/bin/env python

import argparse

import chainer
from chainer.training import extensions

import fcn


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    gpu = args.gpu
    resume = None  # filename
    max_iter = 100000

    # 1. dataset
    dataset_train = fcn.datasets.PascalVOC2012SegmentationDataset('train')
    dataset_val = fcn.datasets.PascalVOC2012SegmentationDataset('val')

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=1, repeat=False, shuffle=False)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    model = fcn.models.FCN32s()
    model.train = True
    fcn.util.copy_chainermodel(vgg, model)

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=1e-10, momentum=0.99)
    optimizer.setup(model)

    # 4. trainer
    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=gpu)
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out='result')

    trainer.extend(TestModeEvaluator(iter_val, model, device=gpu),
                   trigger=(100, 'iteration'))
    trainer.extend(extensions.snapshot(trigger=(100, 'iteration')))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport([
        'iteration',
        'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'main/iu', 'validation/main/iu',
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if resume:
        chainer.serializers.load_hdf5(resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
