#!/usr/bin/env python

import os.path as osp

import numpy as np
import PIL.Image
import skimage.io
import skimage.transform

from fcn import utils


here = osp.dirname(osp.abspath(__file__))


def test_label_accuracy_score():
    img_file = osp.join(here, '../data/2007_000063.jpg')
    lbl_file = osp.join(here, '../data/2007_000063.png')

    img = skimage.io.imread(img_file)

    lbl_gt = np.array(PIL.Image.open(lbl_file), dtype=np.int32, copy=False)
    lbl_gt[lbl_gt == 255] = -1

    lbl_pred = lbl_gt.copy()
    lbl_pred[lbl_pred == -1] = 0
    lbl_pred = skimage.transform.rescale(lbl_pred, 1/16., order=0,
                                         preserve_range=True)
    lbl_pred = skimage.transform.resize(lbl_pred, lbl_gt.shape, order=0,
                                        preserve_range=True)
    lbl_pred = lbl_pred.astype(lbl_gt.dtype)

    acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(
        [lbl_gt], [lbl_gt], n_class=21)
    assert acc == 1
    assert acc_cls == 1
    assert mean_iu == 1
    assert fwavacc == 1

    acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(
        lbl_gt, lbl_pred, n_class=21)
    assert 0.9 <= acc <= 1
    assert 0.9 <= acc_cls <= 1
    assert 0.9 <= mean_iu <= 1
    assert 0.9 <= fwavacc <= 1

    return img, lbl_gt, lbl_pred, acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import skimage.color
    img, lbl_gt, lbl_pred, acc, acc_cls, mean_iu, fwavacc = \
        test_label_accuracy_score()
    print('acc: %.4f' % acc)
    print('acc_cls: %.4f' % acc_cls)
    print('mean_iu: %.4f' % mean_iu)
    print('fwavacc: %.4f' % fwavacc)
    viz_gt = skimage.color.label2rgb(lbl_gt, img)
    viz_pred = skimage.color.label2rgb(lbl_pred, img)
    plt.subplot(121)
    plt.imshow(viz_gt)
    plt.subplot(122)
    plt.imshow(viz_pred)
    plt.show()
