#!/usr/bin/env python

import os.path as osp

import chainer

import fcn


def main():
    dataset_dir = chainer.dataset.get_dataset_directory('apc2016')
    path = osp.join(dataset_dir, 'APC2016rbo.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vSV9oLTd1U2I3TDg',
        path=path,
        md5='efd7f1d5420636ee2b2827e7e0f5d1ac',
    )
    fcn.utils.extract_file(path, to_directory=dataset_dir)

    path = osp.join(dataset_dir, 'APC2016jsk.tgz')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vaExFU1AxWHlMdTg',
        path=path,
        md5='791af8d7d296ed23c3a6690f894cd7f8',
    )
    fcn.utils.extract_file(path, to_directory=dataset_dir)

    # XXX: this is optional
    # path = osp.join(dataset_dir, 'APC2016mit_training.zip')
    # fcn.data.cached_download(
    #     url='https://drive.google.com/uc?id=0B4mCa-2YGnp7ZEMwcW5rcVBpeG8',
    #     path=path,
    # )
    path = osp.join(dataset_dir, 'APC2016mit_benchmark.zip')
    fcn.data.cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZHlSQjJSV0x4eXc',
        path=path,
        md5='15a6ff714fafb3950c0b0ff0161d6ef6',
    )


if __name__ == '__main__':
    main()
