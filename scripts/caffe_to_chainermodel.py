#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path as osp

import caffe
import chainer.functions as F
import chainer.serializers as S

import fcn
from fcn.models import FCN8s


data_dir = fcn.get_data_dir()
caffemodel = osp.join(data_dir, 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel')
caffe_prototxt = osp.join(data_dir, 'voc-fcn8s/deploy.prototxt')
chainermodel = osp.join(data_dir, 'fcn8s.chainermodel')

net = caffe.Net(caffe_prototxt, caffemodel, caffe.TEST)

# TODO(pfnet): chainer CaffeFunction not support some layers
# from chainer.functions.caffe import CaffeFunction
# func = CaffeFunction(caffemodel)

model = FCN8s()
for name, param in net.params.iteritems():
    layer = getattr(model, name)

    has_bias = True
    if len(param) == 1:
        has_bias = False

    print('{0}:'.format(name))
    # weight
    print('  - W:', param[0].data.shape, layer.W.data.shape)
    assert param[0].data.shape == layer.W.data.shape
    layer.W.data = param[0].data
    # bias
    if has_bias:
        print('  - b:', param[1].data.shape, layer.b.data.shape)
        assert param[1].data.shape == layer.b.data.shape
        layer.b.data = param[1].data

S.save_hdf5(chainermodel, model)
