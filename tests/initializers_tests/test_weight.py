#!/usr/bin/env python

import os.path as osp

import chainer
import numpy as np
import skimage.io

from fcn import initializers


here = osp.dirname(osp.abspath(__file__))


def test_get_upsampling_filter():
    img_file = osp.join(here, '../data/2007_000063.jpg')
    src = skimage.io.imread(img_file)

    c1 = 3
    c2 = 3
    ksize = 4

    filt = initializers.weight._get_upsampling_filter(ksize)
    link = chainer.links.Deconvolution2D(
        c1, c2, ksize, stride=2, pad=0, nobias=True)
    link.W.data[...] = 0
    link.W.data[range(c1), range(c2), :, :] = filt

    input = src.astype(np.float32)
    input = input.transpose(2, 0, 1)
    input = input[np.newaxis, :, :, :]
    input = chainer.Variable(input)

    output = link(input)

    dst = output.data[0]
    dst = dst.transpose(1, 2, 0)
    dst = dst.astype(np.uint8)

    assert dst.shape == (752, 1002, 3)

    return src, dst


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    src, dst = test_get_upsampling_filter()
    plt.subplot(121)
    plt.imshow(src)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()
