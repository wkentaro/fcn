from __future__ import print_function

import os
import os.path as osp
import sys
import tempfile

import chainer
from chainer.training import extensions
import numpy as np
import skimage.color
import yaml

import fcn


def get_trainer(
        dataset_class,
        gpu,
        max_iter,
        out=None,
        resume=None,
        interval_log=10,
        interval_eval=1000,
        optimizer=None,
        batch_size=1,
        ):

    if isinstance(gpu, list):
        gpus = gpu
    else:
        gpus = [gpu]

    if out is None:
        out = tempfile.mktemp()

    if optimizer is None:
        optimizer = chainer.optimizers.MomentumSGD(lr=1e-10, momentum=0.99)

    if not resume and osp.exists(out):
        print('Result dir already exists: {}'.format(osp.abspath(out)),
              file=sys.stderr)
        quit(1)

    # dump params
    params = {
        'dataset_class': dataset_class.__name__,
        'gpu': {'ids': gpu},
        'batch_size': batch_size,
        'resume': resume,
        'optimizer': {
            'name': optimizer.__class__.__name__,
            'params': optimizer.__dict__,
        }
    }
    print('>' * 20 + ' Parameters ' + '>' * 20)
    yaml.safe_dump(params, sys.stderr, default_flow_style=False)
    print('<' * 20 + ' Parameters ' + '<' * 20)
    if not osp.exists(out):
        os.makedirs(out)
    yaml.safe_dump(params, open(osp.join(out, 'param.yaml'), 'w'),
                   default_flow_style=False)

    # 1. dataset
    dataset_train = dataset_class('train')
    dataset_val = dataset_class('val')

    if len(gpus) > 1:
        iter_train = chainer.iterators.MultiprocessIterator(
            dataset_train, batch_size=batch_size*len(gpus),
            shared_mem=10000000)
    else:
        iter_train = chainer.iterators.SerialIterator(
            dataset_train, batch_size=batch_size)
    iter_val = chainer.iterators.SerialIterator(
        dataset_val, batch_size=batch_size, repeat=False, shuffle=False)

    # 2. model
    vgg_path = fcn.data.download_vgg16_chainermodel()
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    n_class = len(dataset_train.label_names)
    model = fcn.models.FCN32s(n_class=n_class)
    model.train = True
    fcn.util.copy_chainermodel(vgg, model)

    if len(gpus) > 1 or gpus[0] >= 0:
        chainer.cuda.get_device(gpus[0]).use()
    if len(gpus) == 1 and gpus[0] >= 0:
        model.to_gpu()

    # 3. optimizer
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # 4. trainer
    if len(gpus) > 1:
        devices = {'main': gpus[0]}
        for gpu in gpus[1:]:
            devices['gpu{}'.format(gpu)] = gpu
        updater = chainer.training.ParallelUpdater(
            iter_train, optimizer, devices=devices)
    else:
        updater = chainer.training.StandardUpdater(
            iter_train, optimizer, device=gpus[0])
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out)

    trainer.extend(
        fcn.training.extensions.TestModeEvaluator(
            iter_val, model, device=gpus[0]),
        trigger=(interval_eval, 'iteration'),
        invoke_before_training=False,
    )

    def visualize_segmentation(target):
        datum = chainer.cuda.to_cpu(target.x.data[0])
        img = dataset_val.datum_to_img(datum)
        label_true = chainer.cuda.to_cpu(target.t.data[0])
        label_pred = chainer.cuda.to_cpu(target.score.data[0]).argmax(axis=0)
        label_pred[label_true == -1] = 0

        cmap = fcn.util.labelcolormap(len(dataset_val.label_names))
        label_viz0 = skimage.color.label2rgb(
            label_pred, colors=cmap[1:], bg_label=0)
        label_viz0[label_true == -1] = (0, 0, 0)
        label_viz0 = (label_viz0 * 255).astype(np.uint8)

        label_viz1 = skimage.color.label2rgb(
            label_pred, img, colors=cmap[1:], bg_label=0)
        label_viz1[label_true == -1] = (0, 0, 0)
        label_viz1 = (label_viz1 * 255).astype(np.uint8)

        return fcn.util.get_tile_image([img, label_viz0, label_viz1])

    trainer.extend(
        fcn.training.extensions.ImageVisualizer(
            iter_val, model, viz_func=visualize_segmentation, device=gpus[0]),
        trigger=(interval_eval, 'iteration'),
        invoke_before_training=True,
    )

    model_name = model.__class__.__name__
    trainer.extend(extensions.snapshot(
        savefun=chainer.serializers.hdf5.save_hdf5,
        filename='%s_trainer_iter_{.updater.iteration}.h5' % model_name,
        trigger=(interval_eval, 'iteration')))
    trainer.extend(extensions.snapshot_object(
        model,
        savefun=chainer.serializers.hdf5.save_hdf5,
        filename='%s_model_iter_{.updater.iteration}.h5' % model_name,
        trigger=(interval_eval, 'iteration')))
    trainer.extend(extensions.LogReport(
        trigger=(interval_log, 'iteration'), log_name='log.json'))
    trainer.extend(extensions.PrintReport([
        'iteration',
        'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'main/iu', 'validation/main/iu',
        'elapsed_time',
    ]))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if resume:
        if resume.endswith('npz'):
            chainer.serializers.load_npz(resume, trainer)
        else:
            chainer.serializers.load_hdf5(resume, trainer)

    return trainer
