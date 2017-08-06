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

    viz = utils.visualize_segmentation(
        lbl_pred=lbl_pred, img=img, n_class=21, lbl_true=lbl_gt)

    img_h, img_w = img.shape[:2]

    assert isinstance(viz, np.ndarray)
    assert viz.shape == (img_h * 2, img_w * 3, 3)
    assert viz.dtype == np.uint8

    return viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import skimage.color
    viz = test_label_accuracy_score()
    plt.imshow(viz)
    plt.show()
