#!/usr/bin/env python

from __future__ import print_function

import os
import os.path as osp
import sys

try:
    import caffe
except ImportError:
    print('Cannot import caffe. Please install it.')
    quit(1)
import chainer.serializers as S

import fcn


here = osp.dirname(osp.abspath(__file__))


sys.path.insert(0, osp.join(here, '../../fcn/external/fcn.berkeleyvision.org'))


def caffe_to_chainermodel(model, caffe_prototxt, caffemodel_path,
                          chainermodel_path):
    os.chdir(osp.dirname(caffe_prototxt))
    net = caffe.Net(caffe_prototxt, caffemodel_path, caffe.TEST)

    for name, param in net.params.iteritems():
        try:
            layer = getattr(model, name)
        except AttributeError:
            print('Skipping caffe layer: %s' % name)
            continue

        has_bias = True
        if len(param) == 1:
            has_bias = False

        print('{0}:'.format(name))
        # weight
        print('  - W: %s %s' % (param[0].data.shape, layer.W.data.shape))
        assert param[0].data.shape == layer.W.data.shape
        layer.W.data = param[0].data
        # bias
        if has_bias:
            print('  - b: %s %s' % (param[1].data.shape, layer.b.data.shape))
            assert param[1].data.shape == layer.b.data.shape
            layer.b.data = param[1].data
    S.save_npz(chainermodel_path, model)


def main():
    for model_name in ['FCN8s', 'FCN8sAtOnce', 'FCN16s', 'FCN32s']:
        print('[caffe_to_chainermodel.py] converting model: %s' % model_name)
        # get model
        model = getattr(fcn.models, model_name)()
        if model_name == 'FCN8sAtOnce':
            model_name = 'fcn8s-atonce'
        else:
            model_name = model_name.lower()

        # get caffemodel
        caffe_prototxt = osp.join(
            here, '../..',
            'fcn/external/fcn.berkeleyvision.org/voc-%s/deploy.prototxt' %
            model_name)
        caffemodel = osp.expanduser(
            '~/data/models/caffe/%s-heavy-pascal.caffemodel' % model_name)
        if not osp.exists(caffemodel):
            file = osp.join(osp.dirname(caffe_prototxt), 'caffemodel-url')
            url = open(file).read().strip()
            fcn.data.cached_download(url, caffemodel)

        # convert caffemodel to chainermodel
        chainermodel = osp.expanduser(
            '~/data/models/chainer/%s_from_caffe.npz' % model_name)
        if not osp.exists(chainermodel):
            caffe_to_chainermodel(model, caffe_prototxt,
                                  caffemodel, chainermodel)


if __name__ == '__main__':
    main()
