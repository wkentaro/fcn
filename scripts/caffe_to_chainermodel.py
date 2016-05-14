#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import os.path as osp
import shlex
import subprocess
import sys

import caffe
import chainer.functions as F
import chainer.serializers as S

import fcn
from fcn.models import FCN8s


def main():
    chainermodel = osp.join(fcn.data_dir, 'fcn8s_from_caffe.chainermodel')
    md5 = 'a1083db5a47643b112af69bfa59954f9'
    print("Checking md5: '{0}' for '{1}'".format(md5, chainermodel))
    if osp.exists(chainermodel) and fcn.util.check_md5(chainermodel, md5):
        print("'{0}' is already newest version.".format(chainermodel))
        sys.exit(0)

    caffe_prototxt = osp.join(
        fcn.data_dir, 'fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt')
    caffemodel = fcn.setup.download_fcn8s_caffemodel()
    net = caffe.Net(caffe_prototxt, caffemodel, caffe.TEST)

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


if __name__ == '__main__':
    main()
