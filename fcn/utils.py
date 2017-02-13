from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cStringIO as StringIO
import hashlib
import json
import math
import os
import os.path as osp
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# CV Util
# -----------------------------------------------------------------------------

def apply_mask(img, mask, crop=False, fill_black=True):
    if fill_black:
        img[mask == 0] = 0

    if crop:
        where = np.argwhere(mask)
        (y_start, x_start), (y_stop, x_stop) = where.min(0), where.max(0) + 1
        img = img[y_start:y_stop, x_start:x_stop]

    return img


def resize_img_with_max_size(img, max_size=500*500):
    """Resize image with max size (height x width)"""
    from skimage.transform import rescale
    height, width = img.shape[:2]
    scale = max_size / (height * width)
    resizing_scale = 1
    if scale < 1:
        resizing_scale = np.sqrt(scale)
        img = rescale(img, resizing_scale, preserve_range=True)
        img = img.astype(np.uint8)
    return img, resizing_scale


# -----------------------------------------------------------------------------
# Chainer Util
# -----------------------------------------------------------------------------

def copy_chainermodel(src, dst):
    from chainer import link
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    print('Copying layers %s -> %s:' %
          (src.__class__.__name__, dst.__class__.__name__))
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_chainermodel(child, dst_child)
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
                print('Ignore %s because of parameter mismatch.' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print(' layer: %s -> %s' % (child.name, dst_child.name))


def draw_computational_graph(*args, **kwargs):
    """Draw computational graph.

    @param output: output ps file.
    """
    from chainer.computational_graph import build_computational_graph
    output = kwargs.pop('output')
    if len(args) > 2:
        variable_style = args[2]
    else:
        variable_style = kwargs.get(
            'variable_style',
            {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'},
        )
        kwargs['variable_style'] = variable_style
    if len(args) > 3:
        function_style = args[3]
    else:
        function_style = kwargs.get(
            'function_style',
            {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'},
        )
        kwargs['function_style'] = function_style
    dotfile = tempfile.mktemp()
    with open(dotfile, 'w') as f:
        f.write(build_computational_graph(*args, **kwargs).dump())
    ext = osp.splitext(output)[-1][1:]  # ex) .ps -> ps
    cmd = 'dot -T{0} {1} > {2}'.format(ext, dotfile, output)
    subprocess.call(cmd, shell=True)


def append_log_to_json(log, log_file):
    if osp.exists(log_file):
        logs = json.load(open(log_file))
    else:
        logs = []
    logs.append(log)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4, sort_keys=True)


def batch_to_vars(batch, device=-1, volatile='off'):
    import chainer
    from chainer import cuda
    in_arrays = [np.asarray(x) for x in zip(*batch)]
    if device >= 0:
        in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
    in_vars = [chainer.Variable(x, volatile) for x in in_arrays]
    return in_vars


# -----------------------------------------------------------------------------
# Data Util
# -----------------------------------------------------------------------------


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


def download(client, url, output, quiet=False):
    cmd = '{client} {url} -O {output}'.format(client=client, url=url,
                                              output=output)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))


def check_md5(path, md5):
    is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
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
    print("Checking md5 of '{path}'...".format(path=cache_file))
    # check real path
    if osp.exists(path):
        if check_md5(path, md5):
            print("File '{0}' is newest.".format(path))
            if extract:
                print("Extracting '{path}'...".format(path=path))
                extract_file(path, to_directory=osp.dirname(path))
            return
        else:
            if not osp.islink(path):
                # not link and exists so skipping
                sys.stderr.write("WARNING: '{0}' exists\n".format(path))
                return
            os.remove(path)
    else:
        if osp.islink(path):
            os.remove(path)
    # check cache path
    if osp.exists(cache_file):
        if check_md5(cache_file, md5):
            print("Cache file '{0}' is newest.".format(cache_file))
            os.symlink(cache_file, path)
            if extract:
                print("Extracting '{path}'...".format(path=path))
                extract_file(path, to_directory=osp.dirname(path))
            return
        else:
            os.remove(cache_file)
    print("Downloading file from '{url}'...".format(url=url))
    download(download_client, url, cache_file, quiet=quiet)
    os.symlink(cache_file, path)
    if extract:
        print("Extracting '{path}'...".format(path=path))
        extract_file(path, to_directory=osp.dirname(path))


# -----------------------------------------------------------------------------
# Color Util
# -----------------------------------------------------------------------------

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(N=256):
    cmap = np.zeros((N, 3))
    for i in xrange(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in xrange(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def visualize_labelcolormap(cmap):
    n_colors = len(cmap)
    ret = np.zeros((n_colors, 10 * 10, 3))
    for i in xrange(n_colors):
        ret[i, ...] = cmap[i]
    return ret.reshape((n_colors * 10, 10, 3))


def get_label_colortable(n_labels, shape):
    import cv2
    rows, cols = shape
    if rows * cols < n_labels:
        raise ValueError
    cmap = labelcolormap(n_labels)
    table = np.zeros((rows * cols, 50, 50, 3), dtype=np.uint8)
    for lbl_id, color in enumerate(cmap):
        color_uint8 = (color * 255).astype(np.uint8)
        table[lbl_id, :, :] = color_uint8
        text = '{:<2}'.format(lbl_id)
        cv2.putText(table[lbl_id], text, (5, 35),
                    cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    table = table.reshape(rows, cols, 50, 50, 3)
    table = table.transpose(0, 2, 1, 3, 4)
    table = table.reshape(rows * 50, cols * 50, 3)
    return table


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_true, label_pred, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = _fast_hist(label_true.flatten(), label_pred.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def draw_label(label, img, n_class, label_titles, bg_label=0):
    """Convert label to rgb with label titles.

    @param label_title: label title for each labels.
    @type label_title: dict
    """
    from PIL import Image
    from scipy.misc import fromimage
    from skimage.color import label2rgb
    from skimage.transform import resize
    colors = labelcolormap(n_class)
    label_viz = label2rgb(label, img, colors=colors[1:], bg_label=bg_label)
    # label 0 color: (0, 0, 0, 0) -> (0, 0, 0, 255)
    label_viz[label == 0] = 0

    # plot label titles on image using matplotlib
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    # plot image
    plt.imshow(label_viz)
    # plot legend
    plt_handlers = []
    plt_titles = []
    for label_value in np.unique(label):
        if label_value not in label_titles:
            continue
        fc = colors[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_titles[label_value])
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=0.5)
    # convert plotted figure to np.ndarray
    f = StringIO.StringIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    result_img_pil = Image.open(f)
    result_img = fromimage(result_img_pil, mode='RGB')
    result_img = resize(result_img, img.shape, preserve_range=True)
    result_img = result_img.astype(img.dtype)
    return result_img


def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical+h,
               pad_horizontal:pad_horizontal+w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, 3), dtype=np.uint8)
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y*one_height:(y+1)*one_height,
                                   x*one_width:(x+1)*one_width, ] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    from skimage.transform import resize

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return x_num, y_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(img, (h, w), preserve_range=True).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)


def visualize_segmentation(lbl_pred, lbl_true, img, n_class):
    import skimage.color
    import skimage.util
    lbl_pred[lbl_true == -1] = 0
    lbl_true[lbl_true == -1] = 0

    cmap = labelcolormap(n_class)

    viz_true0 = skimage.color.label2rgb(lbl_true, colors=cmap[1:], bg_label=0)
    viz_true0 = skimage.util.img_as_ubyte(viz_true0)
    viz_true1 = skimage.color.label2rgb(
        lbl_true, img, colors=cmap[1:], bg_label=0)
    viz_true1 = skimage.util.img_as_ubyte(viz_true1)

    viz_pred0 = skimage.color.label2rgb(lbl_pred, colors=cmap[1:], bg_label=0)
    viz_pred0 = skimage.util.img_as_ubyte(viz_pred0)
    viz_pred1 = skimage.color.label2rgb(
        lbl_pred, img, colors=cmap[1:], bg_label=0)
    viz_pred1 = skimage.util.img_as_ubyte(viz_pred1)

    return get_tile_image([viz_true0, viz_true1, viz_pred0, viz_pred1],
                          tile_shape=(2, 2))
