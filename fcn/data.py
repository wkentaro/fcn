import hashlib
import os
import os.path as osp
import tarfile
import zipfile

import gdown


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()


def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5, quiet=False):
        if not quiet:
            print('Checking md5 of file: {}'.format(path))
        return md5sum(path) == md5

    if osp.exists(path) and not md5:
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return gdown.download(url, path, quiet=quiet)


def extract_file(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar'):
        opener, mode = tarfile.open, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError("Could not extract '%s' as no appropriate "
                         "extractor is found" % path)

    cwd = os.getcwd()
    os.chdir(to_directory)
    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)


# Model Download
# -----------------------------------------------------------------------------

MODELS_DIR = osp.expanduser('~/data/models/chainer')


def download_vgg16_chainermodel(check_md5=True):
    path = osp.join(MODELS_DIR, 'vgg16_from_caffe.npz')
    md5 = '54a0cddc1392ccc4056bbeecbb30f3d4' if check_md5 else None
    return cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vRy1XYnRSa1hNSW8',
        path=path,
        md5=md5,
    )


def download_fcn8s_chainermodel(check_md5=True):
    path = osp.join(MODELS_DIR, 'fcn8s_from_caffe.npz')
    md5 = '256c2a8235c1c65e62e48d3284fbd384' if check_md5 else None
    return cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vb0cxV0VhcG1Lb28',
        path=path,
        md5=md5,
    )


def download_fcn16s_chainermodel(check_md5=True):
    path = osp.join(MODELS_DIR, 'fcn16s_from_caffe.npz')
    md5 = '7c9b50a1a8c6c20d3855d4823bbea61e' if check_md5 else None
    return cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vcnBiXzZTcG9FU3c',
        path=path,
        md5=md5,
    )


def download_fcn32s_chainermodel(check_md5=True):
    path = osp.join(MODELS_DIR, 'fcn32s_from_caffe.npz')
    md5 = 'b7f0a2e66229ccdb099c0a60b432e8cf' if check_md5 else None
    return cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vTElpa1p3WFNDczQ',
        path=path,
        md5=md5,
    )
