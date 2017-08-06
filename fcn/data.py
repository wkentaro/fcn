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

    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return md5sum(path) == md5

    if osp.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, md5sum(path)))
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
