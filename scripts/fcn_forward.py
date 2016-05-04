#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import os.path as osp
import tempfile

from chainer import cuda
import chainer.serializers as S
from chainer import Variable
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from skimage.color import label2rgb
from skimage.transform import rescale
from skimage.transform import resize

import fcn
from fcn.models import FCN8s


def img_to_datum(img):
    datum = img.astype(np.float32)
    datum = datum[:, :, ::-1]  # RGB -> BGR
    datum -= np.array((104.00698793, 116.66876762, 122.67891434))
    datum = datum.transpose((2, 0, 1))
    return datum


class Forwarding(object):

    def __init__(self, gpu):
        self.gpu = gpu

        self.data_dir = fcn.get_data_dir()
        chainermodel = osp.join(self.data_dir, 'fcn8s.chainermodel')

        self.target_names = fcn.pascal.get_target_names()
        self.model = FCN8s(n_class=len(self.target_names))
        S.load_hdf5(chainermodel, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def forward_img_file(self, img_file):
        print('{0}:'.format(osp.realpath(img_file)))
        # setup image
        img = imread(img_file)
        scale =  (500 * 500) / (img.shape[0] * img.shape[1])
        if scale < 1:
            resizing_scale = np.sqrt(scale)
            print(' - resizing_scale: {0}'.format(resizing_scale))
            img = rescale(img, resizing_scale, preserve_range=True)
            img = img.astype(np.uint8)
        # setup input datum
        datum = img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data, volatile=True)
        # forward
        self.model.train = False
        pred = self.model(x)
        pred_datum = cuda.to_cpu(pred.data)[0]
        label = np.argmax(pred_datum, axis=0)
        unique_labels = np.unique(label)
        print(' - unique_labels:', unique_labels)
        print(' - target_names:', self.target_names[unique_labels])
        # visualize
        cmap = fcn.util.labelcolormap(21)
        label_viz = label2rgb(label, img, colors=cmap[1:], bg_label=0)
        label_viz[label == 0] = cmap[0]
        # plot image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                            wspace=0, hspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.imshow(label_viz)
        # plot legend
        plt_handlers = []
        plt_titles = []
        for l in np.unique(label):
            fc = cmap[l]
            p = plt.Rectangle((0, 0), 1, 1, fc=fc)
            plt_handlers.append(p)
            plt_titles.append(self.target_names[l])
        plt.legend(plt_handlers, plt_titles)
        result_file = osp.join(tempfile.mkdtemp(), 'result.png')
        plt.savefig(result_file, bbox_inches='tight', pad_inches=0)
        # compose result
        result_img = imread(result_file)
        result_img = resize(result_img, img.shape, preserve_range=True)
        result_img = result_img.astype(np.uint8)
        # save result
        if self.save_dir is None:
            save_dir = 'forward_out'
        else:
            save_dir = osp.join(self.data_dir, 'forward_out')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        height, width = img.shape[:2]
        if height > width:
            vline = np.ones((height, 3, 3), dtype=np.uint8) * 255
            out_img = np.hstack((img, vline, result_img))
        else:
            hline = np.ones((3, width, 3), dtype=np.uint8) * 255
            out_img = np.vstack((img, hline, result_img))
        out_file = osp.join(save_dir, osp.basename(img_file))
        imsave(out_file, out_img)
        print(' - out_file: {0}'.format(out_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only')
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    args = parser.parse_args()

    img_files = args.img_files
    gpu = args.gpu

    forwarding = Forwarding(gpu)
    for img_file in img_files:
        forwarding.forward_img_file(img_file)


if __name__ == '__main__':
    main()
