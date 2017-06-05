#!/usr/bin/env python

import os.path as osp

import chainer

import fcn


def main():
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
    dataset_dir = osp.expanduser('~/data/datasets/VOC')
    path = osp.join(dataset_dir, 'VOCtrainval_11-May-2012.tar')
    fcn.data.cached_download(
        url,
        path=path,
        md5='6cd6e144f989b92b3379bac3b3de84fd',
    )
    fcn.utils.extract_file(path, to_directory=dataset_dir)


if __name__ == '__main__':
    main()
