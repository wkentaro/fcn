import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

from fcn import utils


class FCN32s(chainer.Chain):

    """Full Convolutional Network 32s"""

    def __init__(self, n_class=21, nodeconv=False):
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

        )
        if nodeconv:
            self.add_link('upscore',
                          L.Deconvolution2D(self.n_class, self.n_class, 64,
                                            stride=32, pad=0))
        else:
            self.upscore = None
        self.train = False

    def __call__(self, x, t=None):
        self.data = cuda.to_cpu(x.data)

        # conv1
        h = F.relu(self.conv1_1(x), use_cudnn=False)
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1), use_cudnn=False)
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1), use_cudnn=False)
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1), use_cudnn=False)
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2), use_cudnn=False)
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1), use_cudnn=False)
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2), use_cudnn=False)
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3), use_cudnn=False)
        h = F.relu(self.conv4_2(h), use_cudnn=False)
        h = F.relu(self.conv4_3(h), use_cudnn=False)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4), use_cudnn=False)
        h = F.relu(self.conv5_2(h), use_cudnn=False)
        h = F.relu(self.conv5_3(h), use_cudnn=False)
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # fc6
        h = F.relu(self.fc6(pool5), use_cudnn=False)
        h = F.dropout(h, ratio=.5, train=self.train)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6), use_cudnn=False)
        h = F.dropout(h, ratio=.5, train=self.train)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # upscore
        if self.upscore:
            h = self.upscore(score_fr)
        else:
            in_h, in_w = score_fr.shape[2:4]
            out_h = chainer.utils.conv.get_deconv_outsize(
                in_h, k=32, s=32, p=0)
            out_w = chainer.utils.conv.get_deconv_outsize(
                in_w, k=32, s=32, p=0)
            h = F.resize_images(score_fr, (out_h, out_w))
        upscore = h  # 1

        # score
        h = upscore[:, :, 19:19+x.data.shape[2], 19:19+x.data.shape[3]]
        score = h
        self.score = score  # XXX: for backward compatibility
        # self.score = cuda.to_cpu(h.data)

        if t is None:
            assert not self.train
            return

        # testing with t or training
        loss = F.softmax_cross_entropy(self.score, t, normalize=False)
        self.loss = float(cuda.to_cpu(loss.data))
        if np.isnan(self.loss):
            raise ValueError('loss value is nan')

        self.lbl_true = chainer.cuda.to_cpu(t.data)
        self.lbl_pred = cuda.to_cpu(score.data).argmax(axis=1)

        logs = []
        for i in xrange(x.shape[0]):
            acc, acc_cls, iu, fwavacc = utils.label_accuracy_score(
                self.lbl_true[i], self.lbl_pred[i], self.n_class)
            logs.append((acc, acc_cls, iu, fwavacc))
        log = np.array(logs).mean(axis=0)
        self.log = {
            'loss': self.loss,
            'acc': log[0],
            'acc_cls': log[1],
            'mean_iu': log[2],
            'fwavacc': log[3],
        }
        chainer.report(self.log, self)

        return loss

    def init_from_vgg16(self, vgg16, copy_fc8=True, init_upscore=True):
        for l in self.children():
            if l.name.startswith('conv'):
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data = l1.W.data
                l2.b.data = l1.b.data
            elif l.name in ['fc6', 'fc7']:
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data = l1.W.data.reshape(l2.W.shape)
                l2.b.data = l1.b.data.reshape(l2.b.shape)
            elif l.name == 'score_fr' and copy_fc8:
                l1 = getattr(vgg16, 'fc8')
                l2 = getattr(self, 'score_fr')
                W = l1.W.data[:self.n_class, :]
                b = l1.b.data[:self.n_class]
                assert W.size == l2.W.size
                assert b.size == l2.b.size
                l2.W.data = W.reshape(l2.W.shape)
                l2.b.data = b.reshape(l2.b.shape)
            elif l.name == 'upscore' and init_upscore:
                l2 = getattr(self, 'upscore')
                c1, c2, h, w = l2.W.data.shape
                assert c1 == c2 == self.n_class
                assert h == w
                W = utils.get_upsample_filter(h)
                l2.W.data = np.tile(W.reshape((1, 1, h, w)), (c1, c2, 1, 1))
