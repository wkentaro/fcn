import copy
import os
import os.path as osp
import shutil

import chainer
from chainer import training
import numpy as np
import skimage.io

import fcn


class SemanticSegmentationVisReport(training.Extension):

    def __init__(self, pred_func, iterator, transform, class_names,
                 converter=chainer.dataset.concat_examples,
                 device=None, shape=(3, 3)):
        self.pred_func = pred_func
        self._iterator = iterator
        self._transform = transform
        self._class_names = class_names
        self.converter = converter
        self.device = device
        self._shape = shape

    def __call__(self, trainer):
        try:
            os.makedirs(osp.join(trainer.out, 'visualizations'))
        except OSError:
            pass

        iterator = self._iterator
        it = copy.deepcopy(iterator)

        vizs = []
        for batch in it:
            img, lbl_true = zip(*batch)
            batch = list(map(self._transform, batch))
            x = trainer.updater.converter(batch, self.device)[0]
            with chainer.using_config('enable_backprop', False), \
                    chainer.using_config('train', False):
                score = self.pred_func(x)
                lbl_pred = chainer.functions.argmax(score, axis=1)
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.array)

            batch_size = len(batch)
            for i in range(batch_size):
                im = img[i]
                lt = lbl_true[i]
                lp = lbl_pred[i]
                lp[lt == -1] = -1
                lt = fcn.utils.label2rgb(
                    lt, im, label_names=self._class_names)
                lp = fcn.utils.label2rgb(
                    lp, im, label_names=self._class_names)
                viz = np.hstack([im, lt, lp])
                vizs.append(viz)
                if len(vizs) >= (self._shape[0] * self._shape[1]):
                    break
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = fcn.utils.get_tile_image(vizs, tile_shape=self._shape)
        out_file = osp.join(trainer.out, 'visualizations',
                            '%08d.jpg' % trainer.updater.iteration)
        skimage.io.imsave(out_file, viz)
        out_latest_file = osp.join(trainer.out, 'visualizations/latest.jpg')
        shutil.copy(out_file, out_latest_file)
