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


def download_vgg16():
    path = osp.join(fcn.data_dir, 'vgg16.chainermodel')
    fcn.util.download_data(
        pkg_name='fcn',
        path=path,
        url='https://www.dropbox.com/s/oubwxgmqzep24yq/VGG.model?dl=0',
        md5='292e6472062392f5de02ef431bba4a48',
    )
    return path
