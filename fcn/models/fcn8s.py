import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from fcn import utils


class FCN8s(chainer.Chain):

    """Full Convolutional Network 8s"""

    def __init__(self, n_class=21):
        self.n_class = n_class
        super(FCN8s, self).__init__(
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

            score_fr=L.Convolution2D(4096, n_class, 1, stride=1, pad=0),

            upscore2=L.Deconvolution2D(n_class, n_class, 4, stride=2,
                                       pad=0, nobias=True),
            upscore8=L.Deconvolution2D(n_class, n_class, 16, stride=8,
                                       pad=0, nobias=True),

            score_pool3=L.Convolution2D(256, n_class, 1, stride=1, pad=0),
            score_pool4=L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            upscore_pool4=L.Deconvolution2D(n_class, n_class, 4, stride=2,
                                            pad=0, nobias=True),
        )
        # initialize weights for deconv layer
        filt = utils.get_upsampling_filter(4)
        self.upscore2.W.data[...] = 0
        self.upscore2.W.data[range(n_class), range(n_class), :, :] = filt
        filt = utils.get_upsampling_filter(16)
        self.upscore8.W.data[...] = 0
        self.upscore8.W.data[range(n_class), range(n_class), :, :] = filt
        filt = utils.get_upsampling_filter(4)
        self.upscore_pool4.W.data[...] = 0
        self.upscore_pool4.W.data[range(n_class), range(n_class), :, :] = filt

    def __call__(self, x, t=None):
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
        h = F.dropout(h, ratio=.5)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # score_pool3
        h = self.score_pool3(pool3)
        score_pool3 = h  # 1/8

        # score_pool4
        h = self.score_pool4(pool4)
        score_pool4 = h  # 1/16

        # upscore2
        h = self.upscore2(score_fr)
        upscore2 = h  # 1/16

        # score_pool4c
        h = score_pool4[:, :,
                        5:5+upscore2.data.shape[2], 5:5+upscore2.data.shape[3]]
        score_pool4c = h  # 1/16

        # fuse_pool4
        h = upscore2 + score_pool4c
        fuse_pool4 = h  # 1/16

        # upscore_pool4
        h = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = h  # 1/8

        # score_pool4c
        h = score_pool3[:, :,
                        9:9+upscore_pool4.data.shape[2],
                        9:9+upscore_pool4.data.shape[3]]
        score_pool3c = h  # 1/8

        # fuse_pool3
        h = upscore_pool4 + score_pool3c
        fuse_pool3 = h  # 1/8

        # upscore8
        h = self.upscore8(fuse_pool3)
        upscore8 = h  # 1/1

        # score
        h = upscore8[:, :, 31:31+x.data.shape[2], 31:31+x.data.shape[3]]
        score = h  # 1/1
        self.score = score

        if t is None:
            return

        loss = F.softmax_cross_entropy(score, t, normalize=False)
        if np.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')
        return loss

    def init_from_fcn16s(self, fcn16s):
        for l2 in self.children():
            if l2.name.startswith('up'):
                continue
            try:
                l1 = getattr(fcn16s, l2.name)
            except Exception:
                continue
            assert l1.W.shape == l2.W.shape
            l2.W.data = l1.W.data
            if l2.b is not None:
                assert l1.b.shape == l2.b.shape
                l2.b.data = l1.b.data
