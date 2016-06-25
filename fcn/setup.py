from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import fcn


def _get_data_dir():
    this_dir = osp.dirname(osp.abspath(__file__))
    # 1. python setup.py develop
    data_dir = osp.realpath(osp.join(this_dir, '_data'))
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


data_dir = _get_data_dir()


def download_vgg16_chainermodel():
    path = osp.join(data_dir, 'vgg16.chainermodel')
    fcn.util.download_data(
        pkg_name='fcn',
        path=path,
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vSlFjQlJFQjM5TEk',
        md5='292e6472062392f5de02ef431bba4a48',
    )
    return path


def download_fcn8s_caffemodel():
    caffemodel_dir = osp.join(data_dir, 'fcn.berkeleyvision.org/voc-fcn8s')
    caffemodel = osp.join(caffemodel_dir, 'fcn8s-heavy-pascal.caffemodel')
    url_file = osp.join(caffemodel_dir, 'caffemodel-url')
    fcn.util.download_data(
        pkg_name='fcn',
        path=caffemodel,
        url=open(url_file).read().strip(),
        md5='c03b2953ebd846c270da1a8e8f200c09',
    )
    return caffemodel


class FCN8sFromCaffeChainerModel(object):

    path = osp.join(data_dir, 'fcn8s_from_caffe.chainermodel')
    url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vTXU0QzUwSkVwOFk'
    md5 = 'a1083db5a47643b112af69bfa59954f9'

    def exists(self):
        return (osp.exists(self.path) and
                fcn.util.check_md5(self.path, self.md5))

    def download(self):
        fcn.util.download_data(
            pkg_name='fcn',
            path=self.path,
            url=self.url,
            md5=self.md5
        )
        return self.path
