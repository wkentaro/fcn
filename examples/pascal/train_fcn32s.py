#!/usr/bin/env python

import datetime
import os
import os.path as osp
import sys

import chainer
from chainer import cuda
import click
import pytz
import yaml

import fcn
from fcn.datasets import PascalVOC2012SegmentationDataset


this_dir = osp.dirname(osp.abspath(__file__))


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('-g', '--gpu', type=int, required=True)
def main(gpu, config_file):
    # 0. config

    config = yaml.load(open(config_file))

    out = osp.splitext(osp.basename(config_file))[0]
    for key, value in sorted(config.items()):
        if key == 'name':
            continue
        if isinstance(value, basestring):
            value = value.replace('/', 'SLASH')
            value = value.replace(':', 'COLON')
        out += '_{key}-{value}'.format(key=key.upper(), value=value)
    config['out'] = osp.join(this_dir, 'logs', config['name'], out)

    config['config_file'] = osp.realpath(config_file)
    config['timestamp'] = datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).isoformat()
    if not osp.exists(config['out']):
        os.makedirs(config['out'])
    with open(osp.join(config['out'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    yaml.safe_dump(config, sys.stderr, default_flow_style=False)

    # 1. dataset

    dataset_train = PascalVOC2012SegmentationDataset('train')
    dataset_val = PascalVOC2012SegmentationDataset('val')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=1, shared_mem=10**7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_val, batch_size=1, shared_mem=10**7,
        repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.label_names)

    vgg_path = osp.join(chainer.dataset.get_dataset_directory('fcn'),
                        'vgg16.chainermodel')
    vgg = fcn.models.VGG16()
    chainer.serializers.load_hdf5(vgg_path, vgg)

    model = fcn.models.FCN32s(n_class=n_class)
    model.init_from_vgg16(vgg, copy_fc8=True, init_upscore=False)
    model.train = True

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=config['lr'])
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # training loop

    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_valid=iter_valid,
        out=config['out'],
    )
    trainer.train(max_iter=config['max_iteration'])


if __name__ == '__main__':
    main()
