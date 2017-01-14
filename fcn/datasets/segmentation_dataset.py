#!/usr/bin/env python

import chainer
import numpy as np
import skimage.color

import fcn


class SegmentationDatasetBase(chainer.dataset.DatasetMixin):

    label_names = None
    mean_bgr = None

    def visualize_example(self, i):
        datum, label = self.get_example(i)
        img = self.datum_to_img(datum)
        cmap = fcn.util.labelcolormap(len(self.label_names))
        ignore_mask = [label == -1]
        label[ignore_mask] = 0
        labelviz = skimage.color.label2rgb(
            label, image=img, colors=cmap[1:], bg_label=0)
        labelviz[ignore_mask] = (0, 0, 0)
        labelviz = (labelviz * 255).astype(np.uint8)
        return labelviz

    def label_rgb_to_32sc1(self, label_rgb):
        assert label_rgb.dtype == np.uint8
        label = np.zeros(label_rgb.shape[:2], dtype=np.int32)
        label.fill(-1)
        cmap = fcn.util.labelcolormap(len(self.label_names))
        cmap = (cmap * 255).astype(np.uint8)
        for l, rgb in enumerate(cmap):
            mask = np.all(label_rgb == rgb, axis=-1)
            label[mask] = l
        return label

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1]  # RGB -> BGR
        datum -= self.mean_bgr
        datum = datum.transpose((2, 0, 1))
        return datum

    def datum_to_img(self, datum):
        datum = datum.copy()
        bgr = datum.transpose((1, 2, 0))
        bgr += self.mean_bgr
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.astype(np.uint8)
        return rgb
