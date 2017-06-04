#!/usr/bin/env python

from __future__ import print_function

import os
import os.path as osp
import sys
import pkg_resources

import caffe
import chainer
import chainer.serializers as S
from termcolor import cprint

import fcn
from fcn import models


here = osp.dirname(osp.abspath(__file__))


sys.path.insert(0, osp.join(here, '../../fcn/external/fcn.berkeleyvision.org'))


def caffe_to_chainermodel(model, caffe_prototxt, caffemodel_path,
                          chainermodel_path):
    os.chdir(osp.dirname(caffe_prototxt))
    net = caffe.Net(caffe_prototxt, caffemodel_path, caffe.TEST)

    for name, param in net.params.iteritems():
        layer = getattr(model, name)

        has_bias = True
        if len(param) == 1:
            has_bias = False

        cprint('{0}:'.format(name), color='blue')
        # weight
        cprint('  - W: %s %s' % (param[0].data.shape, layer.W.data.shape),
               color='blue')
        assert param[0].data.shape == layer.W.data.shape
        layer.W.data = param[0].data
        # bias
        if has_bias:
            cprint('  - b: %s %s' % (param[1].data.shape, layer.b.data.shape),
                   color='blue')
            assert param[1].data.shape == layer.b.data.shape
            layer.b.data = param[1].data
    S.save_npz(chainermodel_path, model)


def main():
    for model_name in ['FCN8s', 'FCN16s', 'FCN32s']:
        cprint('[caffe_to_chainermodel.py] converting model: %s' % model_name,
               color='blue')
        # get model
        model = getattr(models, model_name)()
        model_name = model_name.lower()

        # get caffemodel
        caffe_prototxt = osp.join(
            here, '../..',
            'fcn/external/fcn.berkeleyvision.org/voc-%s/deploy.prototxt' %
            model_name)
        caffemodel = osp.expanduser(
            '~/data/models/caffe/%s-heavy-pascal.caffemodel' % model_name)

        # convert caffemodel to chainermodel
        chainermodel = osp.expanduser(
            '~/data/models/chainer/%s_from_caffe.npz' % model_name)
        caffe_to_chainermodel(model, caffe_prototxt, caffemodel, chainermodel)


if __name__ == '__main__':
    main()
