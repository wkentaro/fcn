from __future__ import division

import math
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import scipy.ndimage
import six
import skimage.color


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
# Color Util
# -----------------------------------------------------------------------------

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(*args, **kwargs):
    warnings.warn('labelcolormap is renamed to label_colormap.',
                  DeprecationWarning)
    return label_colormap(*args, **kwargs)


def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def visualize_labelcolormap(*args, **kwargs):
    warnings.warn(
        'visualize_labelcolormap is renamed to visualize_label_colormap',
        DeprecationWarning)
    return visualize_label_colormap(*args, **kwargs)


def visualize_label_colormap(cmap):
    n_colors = len(cmap)
    ret = np.zeros((n_colors, 10 * 10, 3))
    for i in six.moves.range(n_colors):
        ret[i, ...] = cmap[i]
    return ret.reshape((n_colors * 10, 10, 3))


def get_label_colortable(n_labels, shape):
    if cv2 is None:
        raise RuntimeError('get_label_colortable requires OpenCV (cv2)')
    rows, cols = shape
    if rows * cols < n_labels:
        raise ValueError
    cmap = label_colormap(n_labels)
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
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
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
    centerized[pad_vertical:pad_vertical + h,
               pad_horizontal:pad_horizontal + w] = src
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
    for y in six.moves.range(y_num):
        for x in six.moves.range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                                   x * one_width:(x + 1) * one_width] = imgs[i]
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


def label2rgb(lbl, img=None, label_names=None, n_labels=None,
              alpha=0.3, thresh_suppress=0):
    if label_names is None:
        if n_labels is None:
            n_labels = lbl.max() + 1  # +1 for bg_label 0
    else:
        if n_labels is None:
            n_labels = len(label_names)
        else:
            assert n_labels == len(label_names)
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = skimage.color.rgb2gray(img)
        img_gray = skimage.color.gray2rgb(img_gray)
        img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    if label_names is None:
        return lbl_viz

    # cv2 is required only if label_names is not None
    import cv2
    if cv2 is None:
        warnings.warn('label2rgb with label_names requires OpenCV (cv2), '
                      'so ignoring label_names values.')
        return lbl_viz

    np.random.seed(1234)
    for label in np.unique(lbl):
        if label == -1:
            continue  # unlabeled

        mask = lbl == label
        if 1. * mask.sum() / mask.size < thresh_suppress:
            continue
        mask = (mask * 255).astype(np.uint8)
        y, x = scipy.ndimage.center_of_mass(mask)
        y, x = map(int, [y, x])

        if lbl[y, x] != label:
            Y, X = np.where(mask)
            point_index = np.random.randint(0, len(Y))
            y, x = Y[point_index], X[point_index]

        text = label_names[label]
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)

        def get_text_color(color):
            if color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 170:
                return (0, 0, 0)
            return (255, 255, 255)

        color = get_text_color(lbl_viz[y, x])
        cv2.putText(lbl_viz, text,
                    (x - text_size[0] // 2, y),
                    font_face, font_scale, color, thickness)
    return lbl_viz


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
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.

    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred is not None:
            lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            label2rgb(lbl_true, label_names=label_names, n_labels=n_class),
            label2rgb(lbl_true, img, label_names=label_names,
                      n_labels=n_class),
        ]
        viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_trues, (1, 3)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            label2rgb(lbl_pred, label_names=label_names, n_labels=n_class),
            label2rgb(lbl_pred, img, label_names=label_names,
                      n_labels=n_class),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError
