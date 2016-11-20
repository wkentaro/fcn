from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os.path as osp
import re
import shlex
import subprocess

import chainer


def download(url, path, quiet=False):

    def is_google_drive_url(url):
        m = re.match('^https?://drive.google.com/uc\?id=.*$', url)
        return m is not None

    if is_google_drive_url(url):
        client = 'gdown'
    else:
        client = 'wget'

    cmd = '{client} {url} -O {path}'.format(client=client, url=url, path=path)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))

    return path


def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5, quiet=False):
        if not quiet:
            print('Checking md5 of file: {}'.format(path))
        is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
        return is_same

    if osp.exists(path) and not md5:
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return download(url, path, quiet=quiet)


def download_vgg16_chainermodel():
    root = chainer.dataset.get_dataset_directory('fcn')
    path = osp.join(root, 'vgg16.chainermodel')
    return cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vSlFjQlJFQjM5TEk',
        path=path,
        md5='292e6472062392f5de02ef431bba4a48',
    )


def download_fcn8s_caffemodel():
    root = chainer.dataset.get_dataset_directory('fcn')
    path = osp.join(root, 'fcn8s-heavy-pascal.caffemodel')
    return cached_download(
        url='http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel',
        path=path,
        md5='c03b2953ebd846c270da1a8e8f200c09',
    )


def download_fcn8s_from_caffe_chainermodel():
    root = chainer.dataset.get_dataset_directory('fcn')
    path = osp.join(root, 'fcn8s_from_caffe.chainermodel')
    url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vTXU0QzUwSkVwOFk'
    md5 = 'a1083db5a47643b112af69bfa59954f9'
    return cached_download(
        url=url,
        path=path,
        md5=md5,
    )
