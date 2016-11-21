import copy

import six

import chainer
from chainer.training import extension


class Visualizer(extension.Extension):

    trigger = 1, 'epoch'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, viz_func, device=None):
        if isinstance(iterator, chainer.dataset.iterator.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, chainer.link.Link):
            target = {'main': target}
        self._targets = target

        self.device = device
        self.viz_func = viz_func

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

        return self.viz_func(trainer, target)
