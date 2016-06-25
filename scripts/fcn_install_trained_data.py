#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

import fcn
from fcn.util import download_data


def main():
    pkg_name = 'fcn'

    download_data(
        pkg_name=pkg_name,
        path=osp.join(fcn.data_dir, 'fcn8s_from_caffe.chainermodel'),
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vTXU0QzUwSkVwOFk',
        md5='a1083db5a47643b112af69bfa59954f9',
        quiet=False,
    )


if __name__ == '__main__':
    main()
