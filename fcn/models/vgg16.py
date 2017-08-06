import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L

from .. import data


class VGG16(chainer.Chain):

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/vgg16_from_caffe.npz')

    def __init__(self, n_class=1000):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.fc6 = L.Linear(25088, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, n_class)

    def __call__(self, x, t=None):
        h = x
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=.5)
        h = self.fc8(h)
        fc8 = h

        self.proba = F.softmax(fc8)

        if t is None:
            assert not chainer.config.train
            return

        self.loss = F.softmax_cross_entropy(fc8, t)
        self.accuracy = F.accuracy(self.proba, t)
        return self.loss

    @classmethod
    def download(cls):
        return data.cached_download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vRy1XYnRSa1hNSW8',
            path=cls.pretrained_model,
            md5='54a0cddc1392ccc4056bbeecbb30f3d4',
        )
