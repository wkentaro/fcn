#!/usr/bin/env python

from __future__ import print_function

import os.path as osp
import pkg_resources

import caffe
import chainer
import chainer.serializers as S

import fcn
from fcn.models import FCN8s


def fcn8s_caffe_to_chainermodel(caffe_prototxt, caffemodel_path,
                                chainermodel_path):
    net = caffe.Net(caffe_prototxt, caffemodel_path, caffe.TEST)

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
    S.save_hdf5(chainermodel_path, model)


def main():
    # get caffemodel
    pkg_root = pkg_resources.get_distribution('fcn').location
    caffe_prototxt = osp.join(
        pkg_root, 'external/fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt')
    caffemodel = fcn.data.download_fcn8s_caffemodel()

    # convert caffemodel to chainermodel
    chainermodel = osp.join(chainer.dataset.get_dataset_directory('fcn'),
                            'fcn8s_from_caffe.chainermodel')
    fcn8s_caffe_to_chainermodel(caffe_prototxt, caffemodel, chainermodel)


if __name__ == '__main__':
    main()
