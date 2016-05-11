#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from datetime import datetime
import os
import os.path as osp

from chainer import cuda
import chainer.optimizers as O
import chainer.serializers as S
from chainer import Variable
import numpy as np
import tqdm

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
        # setup logging
        self._setup_logging()
        # pretrained model
        pretrained_model = self._setup_pretrained_model()
        # dataset
        self.dataset = pascal.SegmentationClassDataset()
        # setup fcn8s
        self.model = FCN8s(n_class=len(self.dataset.target_names))
        fcn.util.copy_chainermodel(pretrained_model, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        # setup optimizer
        self.optimizer = O.MomentumSGD(lr=1e-12, momentum=0.99)
        self.optimizer.setup(self.model)

    def __del__(self):
        self.logfile.close()

    def _setup_logging(self):
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_dir = osp.join(fcn.get_data_dir(), 'logs', timestamp)
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logfile = open(osp.join(self.log_dir, 'log.csv'), 'w')
        self.logfile.write('i_iter,type,loss,accuracy\n')

    def _setup_pretrained_model(self):
        pretrained_model_path = osp.join(
            fcn.get_data_dir(), 'vgg16.chainermodel')
        md5 = '292e6472062392f5de02ef431bba4a48'
        if not (osp.exists(pretrained_model_path) and
                fcn.util.check_md5(pretrained_model_path, md5)):
            url = 'https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0'
            fcn.util.download_data('fcn', pretrained_model_path, url, md5)
        pretrained_model = VGG16()
        print('Loading pretrained model: {0}'.format(pretrained_model_path))
        S.load_hdf5(pretrained_model_path, pretrained_model)
        return pretrained_model

    def _iterate_once(self, type, indices=None):
        """Iterate with train/val once."""
        batch = self.dataset.next_batch(
            batch_size=1, type=type, indices=indices)
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

    def train(self):
        """Iterate with train data."""
        log_templ = '{i_iter}: type={type}, loss={loss}, accuracy={accuracy}'
        for i_iter in xrange(self.max_iter):
            self.i_iter = i_iter

            if i_iter % self.test_interval == 0:
                self.validate()

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
            self.logfile.write(
                '{i_iter},{type},{loss},{accuracy}\n'.format(**log))

            if i_iter % self.snapshot == 0:
                print('{0}: saving snapshot...'.format(i_iter))
                snapshot_model = osp.join(
                    self.log_dir,
                    'fcn8s_model_{0}.chainermodel'.format(i_iter))
                snapshot_optimizer = osp.join(
                    self.log_dir, 'fcn8s_optimizer_{0}.h5'.format(i_iter))
                S.save_hdf5(snapshot_model, self.model)
                S.save_hdf5(snapshot_optimizer, self.optimizer)

    def validate(self):
        """Validate training with data."""
        log_templ = \
            '{i_iter}: type={type}, mean_loss={loss}, mean_accuracy={accuracy}'
        type = 'val'
        self.model.train = False
        N_data = len(self.dataset.val)
        sum_loss, sum_accuracy = 0, 0
        desc = '{0}: validating'.format(self.i_iter)
        for indice in tqdm.tqdm(xrange(N_data), ncols=80, desc=desc):
            self._iterate_once(type=type, indices=[indice])
            sum_loss += float(self.model.loss.data)
            sum_accuracy += float(self.model.accuracy.data)
        mean_loss = sum_loss / N_data
        mean_accuray = sum_accuracy / N_data
        log = dict(
            i_iter=self.i_iter,
            type=type,
            loss=mean_loss,
            accuracy=mean_accuray,
        )
        print(log_templ.format(**log))
        self.logfile.write('{i_iter},{type},{loss},{accuracy}\n'.format(**log))


if __name__ == '__main__':
    gpu = 0
    trainer = Trainer(
        weight_decay=0.0005,
        test_interval=1000,
        max_iter=100000,
        snapshot=4000,
        gpu=gpu,
    )
    trainer.train()
