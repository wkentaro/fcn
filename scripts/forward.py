#!/usr/bin/env python

import os.path as osp

from chainer import cuda
import chainer.serializers as S
from chainer import Variable
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from skimage.color import label2rgb

import fcn
from fcn.models import FCN8s


def main():
    data_dir = fcn.get_data_dir()

    img_file = osp.join(data_dir, 'pascal/VOC2012/JPEGImages/2007_000129.jpg')
    chainermodel = osp.join(data_dir, 'fcn8s.chainermodel')
    train = False

    model = FCN8s()
    S.load_hdf5(chainermodel, model)
    model.to_gpu()

    img = imread(img_file)
    x_data_0 = img.astype(np.float32)
    x_data_0 = x_data_0[:, :, ::-1]  # RGB -> BGR
    x_data_0 -= np.array((104.00698793, 116.66876762, 122.67891434))
    x_data_0 = x_data_0.transpose((2, 0, 1))
    x_data = np.array([x_data_0], dtype=np.float32)
    x_data = cuda.to_gpu(x_data)
    x = Variable(x_data, volatile=not train)

    model.train = train
    pred = model(x)
    pred_datum = cuda.to_cpu(pred.data)[0]
    label = np.argmax(pred_datum, axis=0)
    print('unique labels:', np.unique(label))
    label_viz = label2rgb(label, img, bg_label=0)

    plt.imshow(label_viz)
    plt.show()


if __name__ == '__main__':
    main()
