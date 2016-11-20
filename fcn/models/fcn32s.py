import math

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import fcn


class FCN32s(chainer.Chain):

    """Full Convolutional Network 32s"""

    def __init__(self, n_class=21):
        self.n_class = n_class
        super(self.__class__, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=100),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Convolution2D(512, 4096, 7, stride=1, pad=0),
            fc7=L.Convolution2D(4096, 4096, 1, stride=1, pad=0),

            score_fr=L.Convolution2D(4096, self.n_class, 1, stride=1, pad=0),

            upscore=L.Deconvolution2D(self.n_class, self.n_class, 64,
                                      stride=32, pad=0),
        )
        self.train = False

    def __call__(self, x, t=None):
        self.x = x
        self.t = t

        # conv1
        h = F.relu(self.conv1_1(x))
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1))
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1))
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1))
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2))
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1))
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2))
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5, train=self.train)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # upscore
        h = self.upscore(score_fr)
        upscore = h  # 1

        # score
        h = upscore[:, :, 19:19+x.data.shape[2], 19:19+x.data.shape[3]]
        self.score = h  # 1/1

        if t is None:
            assert not self.train
            return

        # testing with t or training
        self.loss = F.softmax_cross_entropy(self.score, t, normalize=False)
        if math.isnan(self.loss.data):
            raise ValueError('loss value is nan')

        # report the loss and accuracy
        batch_size = len(x.data)
        labels = chainer.cuda.to_cpu(t.data)
        label_preds = chainer.cuda.to_cpu(self.score.data).argmax(axis=1)
        results = []
        for i in xrange(batch_size):
            acc, acc_cls, iu, fwavacc = fcn.util.label_accuracy_score(
                labels[i], label_preds[i], self.n_class)
            results.append((acc, acc_cls, iu, fwavacc))
        results = np.array(results).mean(axis=0)
        chainer.reporter.report({
            'loss': self.loss,
            'accuracy': results[0],
            'acc_cls': results[1],
            'iu': results[2],
            'fwavacc': results[3],
        }, self)

        return self.loss
