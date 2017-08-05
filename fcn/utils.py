from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cStringIO as StringIO
import hashlib
import math
import os
import os.path as osp
import shlex
import subprocess
import sys
import tarfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Chainer Util
# -----------------------------------------------------------------------------


def batch_to_vars(batch, device=-1):
    import chainer
    from chainer import cuda
    in_arrays = [np.asarray(x) for x in zip(*batch)]
    if device >= 0:
        in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
    in_vars = [chainer.Variable(x) for x in in_arrays]
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
    cmd = 'gdown {url} -O {output}'.format(url=url, output=output)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))


def check_md5(path, md5):
    is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
    return is_same


def download_data(pkg_name, path, url, md5, extract=False, quiet=True):
    """Install test data checking md5 and rosbag decompress if needed."""
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
    download(url, cache_file, quiet=quiet)
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


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
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


def _visualize_segmentation(lbl, n_class=None, img=None, bg_label=0):
    from distutils.version import StrictVersion
    import skimage
    from skimage.color import label2rgb
    from skimage.util import img_as_ubyte
    n_class = len(np.unique(lbl)) if n_class is None else n_class
    cmap = labelcolormap(n_class)
    if StrictVersion(skimage.__version__) <= StrictVersion('0.12.3'):
        colors = cmap[1:]
    else:
        labels = np.unique(lbl)
        labels = labels[labels != bg_label]
        colors = cmap[labels]

    vizs = [img]
    viz0 = label2rgb(lbl, colors=colors, bg_label=bg_label)
    vizs.append(img_as_ubyte(viz0))
    tile_shape = (1, 2)
    if img is not None:
        viz1 = label2rgb(lbl, img, colors=colors, bg_label=bg_label)
        vizs.append(img_as_ubyte(viz1))
        tile_shape = (1, 3)

    return get_tile_image(vizs, tile_shape=tile_shape)


def _visualize_segmentation_legend(label, label_titles, n_class=None,
                                   img=None, bg_label=0):
    """Convert label to rgb with label titles.

    @param label_title: label title for each labels.
    @type label_title: dict
    """
    from PIL import Image
    from scipy.misc import fromimage
    from skimage.transform import resize
    n_class = len(label_titles) if n_class is None else n_class
    colors = labelcolormap(n_class)
    label_viz = _visualize_segmentation(
        label, n_class, img=img, bg_label=bg_label)

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
    result_img = resize(result_img, label_viz.shape, preserve_range=True)
    result_img = result_img.astype(label_viz.dtype)
    return result_img


def visualize_segmentation(**kwargs):
    """Visualize segmentation.

    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict
        Names of each label value. Key is label_value and value is its name.

    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.get('img')
    lbl_true = kwargs.get('lbl_true')
    lbl_pred = kwargs.get('lbl_pred')
    n_class = kwargs.get('n_class')
    label_names = kwargs.get('label_names')

    if lbl_true is not None:
        mask = lbl_true == -1
        lbl_true[mask] = 0
        if lbl_pred is not None:
            lbl_pred[mask] = 0

    vizs = []

    if lbl_true is not None:
        if label_names:
            viz_true = _visualize_segmentation_legend(
                lbl_true, label_names, n_class, img)
        else:
            viz_true = _visualize_segmentation(lbl_true, n_class, img)
        vizs.append(viz_true)

    if lbl_pred is not None:
        if label_names:
            viz_pred = _visualize_segmentation_legend(
                lbl_pred, label_names, n_class, img)
        else:
            viz_pred = _visualize_segmentation(lbl_pred, n_class, img)
        vizs.append(viz_pred)

    img_array = get_tile_image(vizs)
    return img_array
