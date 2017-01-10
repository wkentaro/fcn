import copy
import os.path as osp

import six

import chainer
from chainer.training import extension
import scipy.misc


class ImageVisualizer(extension.Extension):

    trigger = 1, 'epoch'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, viz_func,
                 out='viz_{.updater.iteration}.png', device=None):
        if isinstance(iterator, chainer.dataset.iterator.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, chainer.link.Link):
            target = {'main': target}
        self._targets = target

        self.viz_func = viz_func
        self.out = out
        self.device = device

    def __call__(self, trainer=None):
        iterator = self._iterators['main']
        target = self._targets['main']

        it = copy.copy(iterator)
        batch = next(it)
        in_arrays = chainer.dataset.convert.concat_examples(batch, self.device)

        if isinstance(in_arrays, tuple):
            in_vars = tuple(chainer.Variable(x, volatile='on')
                            for x in in_arrays)
            target(*in_vars)
        elif isinstance(in_arrays, dict):
            in_vars = {key: chainer.Variable(x, volatile='on')
                       for key, x in six.iteritems(in_arrays)}
            target(**in_vars)
        else:
            in_var = chainer.Variable(in_arrays, volatile='on')
            target(in_var)

        result = self.viz_func(target)

        out_path = osp.join(
            trainer.out, self.out.format(trainer))
        scipy.misc.imsave(out_path, result)
