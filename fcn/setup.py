from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

import fcn


def _get_data_dir():
    this_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.realpath(osp.join(this_dir, '_data'))
    if osp.exists(data_dir):
        return data_dir
    return ''


def download_vgg16_chainermodel():
    path = osp.join(fcn.data_dir, 'vgg16.chainermodel')
    fcn.util.download_data(
        pkg_name='fcn',
        path=path,
        url='https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0',
        md5='292e6472062392f5de02ef431bba4a48',
    )
    return path


def download_fcn8s_caffemodel():
    caffemodel_dir = osp.join(fcn.data_dir, 'fcn.berkeleyvision.org/voc-fcn8s')
    caffemodel = osp.join(caffemodel_dir, 'fcn8s-heavy-pascal.caffemodel')
    url_file = osp.join(caffemodel_dir, 'caffemodel-url')
    fcn.util.download_data(
        pkg_name='fcn',
        path=caffemodel,
        url=open(url_file).read().strip(),
        md5 = '4780397b1e1f2ceb98bfa6b03b18dfea',
    )
    return caffemodel
