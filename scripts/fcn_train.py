#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os.path as osp

from chainer import cuda
import chainer.optimizers as O
import chainer.serializers as S
from chainer import Variable
import numpy as np

import fcn
from fcn.models import FCN8s
from fcn.models import VGG16
from fcn import pascal


class Trainer(object):

    def __init__(self, weight_decay, test_interval, max_iter, snapshot, gpu):
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.gpu = gpu
        # pretrained model
        pretrained_model = self._setup_pretrained_model()
        # dataset
        self.dataset = pascal.SegmentationClassDataset()
        # setup fcn8s
        self.model = FCN8s(n_class=len(self.dataset.target_names))
        print('Copying pretrained_model...')
        fcn.util.copy_chainermodel(pretrained_model, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        # setup optimizer
        self.optimizer = O.MomentumSGD(lr=1e-12, momentum=0.99)
        self.optimizer.setup(self.model)

    def _setup_pretrained_model(self):
        data_dir = fcn.get_data_dir()
        pretrained_model_path = osp.join(data_dir, 'vgg16.chainermodel')
        md5 = '292e6472062392f5de02ef431bba4a48'
        if not (osp.exists(pretrained_model_path) and
                fcn.util.check_md5(pretrained_model_path, md5)):
            url = 'https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0'
            fcn.util.download_data('fcn', pretrained_model_path, url, md5)
        pretrained_model = VGG16()
        print('Loading pretrained model: {0}'.format(pretrained_model_path))
        S.load_hdf5(pretrained_model_path, pretrained_model)
        return pretrained_model

    def _iterate_once(self, type):
        """Iterate with train/val once."""
        batch = self.dataset.next_batch(batch_size=1, type=type)
        img, label = batch.img[0], batch.label[0]
        # x
        x_datum = self.dataset.img_to_datum(img)
        x_data = np.array([x_datum], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data, volatile=not self.model.train)
        # y
        y_data = np.array([label], dtype=np.int32)
        if self.gpu != -1:
            y_data = cuda.to_gpu(y_data, device=self.gpu)
        y = Variable(y_data, volatile=not self.model.train)
        # optimize
        if self.model.train:
            self.optimizer.zero_grads()
            self.optimizer.update(self.model, x, y)
        else:
            self.model(x, y)

    def iterate(self):
        """Iterate with train/val data."""
        f = open(osp.join(fcn.get_data_dir(), 'log.csv'), 'w')
        f.write('i_iter,type,loss,accuracy\n')
        log_templ = '{i_iter}: type={type}, loss={loss}, accuracy={accuracy}'
        for i_iter in xrange(self.max_iter):
            #########
            # train #
            #########
            type = 'train'
            self.model.train = True
            self._iterate_once(type=type)
            self.optimizer.weight_decay(self.weight_decay)
            log = dict(
                i_iter=i_iter,
                type=type,
                loss=float(self.model.loss.data),
                accuracy=float(self.model.accuracy.data),
            )
            print(log_templ.format(**log))
            f.write('{i_iter},{type},{loss},{accuracy}\n'.format(**log))

            if i_iter % self.snapshot == 0:
                data_dir = fcn.get_data_dir()
                snapshot_model = osp.join(
                    data_dir, 'fcn8s_model_{0}.chainermodel'.format(i_iter))
                snapshot_optimizer = osp.join(
                    data_dir, 'fcn8s_optimizer_{0}.h5'.format(i_iter))
                S.save_hdf5(snapshot_model, self.model)
                S.save_hdf5(snapshot_optimizer, self.optimizer)

            if i_iter % self.test_interval != 0:
                continue

            ########
            # test #
            ########
            type = 'val'
            self.model.train = False
            self._iterate_once(type=type)
            log = dict(
                i_iter=i_iter,
                type=type,
                loss=float(self.model.loss.data),
                accuracy=float(self.model.accuracy.data),
            )
            print(log_templ.format(**log))
            f.write('{i_iter},{type},{loss},{accuracy}\n'.format(**log))
        f.close()


if __name__ == '__main__':
    gpu = 0
    trainer = Trainer(
        weight_decay=0.0005,
        test_interval=1,
        max_iter=100000,
        snapshot=4000,
        gpu=gpu,
    )
    trainer.iterate()
