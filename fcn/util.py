import hashlib
import os
import os.path as osp
import re
import shlex
import subprocess
import sys
import tarfile
import zipfile


def apply_mask(img, mask, crop=False):
    import numpy as np
    img[mask == 0] = 0

    if crop:
        where = np.argwhere(mask)
        (y_start, x_start), (y_stop, x_stop) = where.min(0), where.max(0) + 1
        img = img[y_start:y_stop, x_start:x_stop]

    return img


def copy_chainermodel(src, dst):
    from chainer import link
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)


def extract_file(path, to_directory='.'):
    print("Extracting '{path}'...".format(path=path))
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
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
    print('...done')


def decompress_rosbag(path, quiet=False):
    print("Decompressing '{path}'...".format(path=path))
    argv = [path]
    if quiet:
        argv.append('--quiet')
    rosbag.rosbag_main.decompress_cmd(argv)
    print('...done')


def download(client, url, output, quiet=False):
    print("Downloading file from '{url}'...".format(url=url))
    cmd = '{client} {url} -O {output}'.format(client=client, url=url,
                                              output=output)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))
    print('...done')


def check_md5sum(path, md5):
    print("Checking md5sum of '{path}'...".format(path=path))
    is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
    print('...done')
    return is_same


def is_google_drive_url(url):
    m = re.match('^https?://drive.google.com/uc\?id=.*$', url)
    return m is not None


def download_data(pkg_name, path, url, md5, download_client=None,
                  extract=False, quiet=True):
    """Install test data checking md5 and rosbag decompress if needed."""
    if download_client is None:
        if is_google_drive_url(url):
            download_client = 'gdown'
        else:
            download_client = 'wget'
    # prepare cache dir
    cache_dir = osp.join(osp.expanduser('~/data'), pkg_name)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = osp.join(cache_dir, osp.basename(path))
    # check if cache exists, and update if necessary
    if not (osp.exists(cache_file) and check_md5sum(cache_file, md5)):
        if osp.exists(cache_file):
            os.remove(cache_file)
        download(download_client, url, cache_file, quiet=quiet)
    if osp.islink(path):
        # overwrite the link
        os.remove(path)
        os.symlink(cache_file, path)
    elif not osp.exists(path):
        if not osp.exists(osp.dirname(path)):
            os.makedirs(osp.dirname(path))
        os.symlink(cache_file, path)  # create link
    else:
        # not link and exists so skipping
        sys.stderr.write("WARNING: '{0}' exists\n".format(path))
        return
    if extract:
        extract_file(path, to_directory=osp.dirname(path))
