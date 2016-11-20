#!/usr/bin/env python

import os.path as osp

import chainer

import fcn.data
import fcn.util


def main():
    url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNnd2RGh1cU5Va28'
    dataset_dir = chainer.dataset.get_dataset_directory('apc2015')
    path = osp.join(dataset_dir, 'APC2015rbo.tgz')
    fcn.data.cached_download(
        url,
        path=path,
        md5='697dde1c5beab563e5ff9a5bb7cd7fc0',
    )
    fcn.util.extract_file(path, to_directory=dataset_dir)


if __name__ == '__main__':
    main()
