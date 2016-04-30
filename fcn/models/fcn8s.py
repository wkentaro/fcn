import math

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np


class FCN8s(chainer.Chain):

    """Full Convolutional Network 8s"""

    def __init__(self, n_class):
        self.n_class = n_class
        super(FCN8s, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
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

            score_pool3=L.Convolution2D(256, self.n_class, 1, stride=1, pad=0),
            score_pool4=L.Convolution2D(512, self.n_class, 1, stride=1, pad=0),
            score_pool5=L.Convolution2D(512, self.n_class, 1, stride=1, pad=0),

            upsample_pool4=L.Deconvolution2D(self.n_class, self.n_class, 4,
                                             stride=2, pad=1, use_cudnn=False),
            upsample_pool5=L.Deconvolution2D(self.n_class, self.n_class, 8,
                                             stride=2, pad=2,
                                             use_cudnn=False),
            upsample_final=L.Deconvolution2D(self.n_class, self.n_class, 16,
                                             stride=2, pad=4,
                                             use_cudnn=False),
        )
        self.train = False

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
        pool3 = h

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        pool4 = h

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        pool5 = h

        p3 = self.score_pool3(pool3)

        p4 = self.score_pool4(pool4)

        p5 = self.score_pool5(pool5)

        self.upsample_pool4.outsize = p3.data.shape[-2:]
        u4 = self.upsample_pool4(p4)

        self.upsample_pool5.outsize = p3.data.shape[-2:]
        u5 = self.upsample_pool5(p5)

        h = p3 + u4 + u5

        self.upsample_final.outsize = x.data.shape[-2:]
        h = self.upsample_final(h)

        # testing without t
        self.pred = F.softmax(h)
        if t is None:
            return self.pred

        # testing with t
        self.accuracy = self.accuracy_score(h, t)
        if not self.train:
            return self.pred

        # training stage
        self.loss = F.softmax_cross_entropy(h, t)
        if np.isnan(cuda.to_cpu(self.loss.data)).sum() != 0:
            raise RuntimeError('ERROR in FCN8s: loss.data contains nan')
        return self.loss

    def accuracy_score(self, y_pred, y_true):
        y_pred = cuda.to_cpu(y_pred.data)
        y_true = cuda.to_cpu(y_true.data)
        # reduce values along classes axis
        reduced_y_pred = np.argmax(y_pred, axis=1)
        s = (reduced_y_pred == y_true).mean()
        return s
