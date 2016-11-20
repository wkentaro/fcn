import os.path as osp
import tempfile

import chainer
from chainer.training import extensions

import fcn


def get_trainer(
        dataset_class,
        gpu,
        max_iter,
        out=None,
        resume=None,
        ):

    if out is None:
        out = tempfile.mktemp()

    if not resume and osp.exists(out):
        raise RuntimeError('Result dir already exists: {}'
                           .format(osp.abspath(out)))

    # 1. dataset
    dataset_train = dataset_class('train')
    dataset_val = dataset_class('val')

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
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # 4. trainer
    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=gpu)
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out)

    trainer.extend(
        fcn.training.extensions.TestModeEvaluator(iter_val, model, device=gpu),
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

    return trainer
