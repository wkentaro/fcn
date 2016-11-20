#!/usr/bin/env python

import os.path as osp

import chainer

import fcn.data
import fcn.util


def main():
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
    dataset_dir = chainer.dataset.get_dataset_directory('pascal')
    path = osp.join(dataset_dir, 'VOCtrainval_11-May-2012.tar')
    fcn.data.cached_download(
        url,
        path=path,
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)


if __name__ == '__main__':
    main()
