#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import os.path as osp
import subprocess
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


class Forwarding(object):

    def __init__(self, gpu):
        self.gpu = gpu

        self.data_dir = fcn.get_data_dir()
        chainermodel = osp.join(self.data_dir, 'fcn8s.chainermodel')

        self.target_names = fcn.pascal.SegmentationClassDataset.target_names
        self.model = FCN8s(n_class=len(self.target_names))
        S.load_hdf5(chainermodel, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def forward_img_file(self, img_file):
        print('{0}:'.format(osp.realpath(img_file)))
        # setup image
        img = imread(img_file, mode='RGB')
        scale = (500 * 500) / (img.shape[0] * img.shape[1])
        if scale < 1:
            resizing_scale = np.sqrt(scale)
            print(' - resizing_scale: {0}'.format(resizing_scale))
            img = rescale(img, resizing_scale, preserve_range=True)
            img = img.astype(np.uint8)
        # setup input datum
        datum = fcn.pascal.SegmentationClassDataset.img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        x = Variable(x_data, volatile=False)
        # forward
        self.model.train = False
        self.model(x)
        pred = self.model.score
        # generate computational_graph
        psfile = osp.join(fcn.get_data_dir(), 'fcn8s_forward.ps')
        if not osp.exists(psfile):
            from chainer.computational_graph import build_computational_graph
            dotfile = tempfile.mktemp()
            with open(dotfile, 'w') as f:
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}
                f.write(build_computational_graph(
                    [pred],
                    variable_style=variable_style,
                    function_style=function_style,
                ).dump())
            cmd = 'dot -Tps {0} > {1}'.format(dotfile, psfile)
            subprocess.call(cmd, shell=True)
            print('- computational_graph: {0}'.format(psfile))
        # visualize result
        pred_datum = cuda.to_cpu(pred.data)[0]
        label = np.argmax(pred_datum, axis=0)
        unique_labels, label_counts = np.unique(label, return_counts=True)
        print('- labels:')
        label_titles = []
        for i, l in enumerate(unique_labels):
            l_region = label_counts[i] / label.size
            title = '{0}:{1} = {2:.1%}'.format(
                l, self.target_names[l], l_region)
            label_titles.append(title)
            print('  - {0}'.format(title))
        # label to rgb
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
        for i, l in enumerate(np.unique(label)):
            fc = cmap[l]
            p = plt.Rectangle((0, 0), 1, 1, fc=fc)
            plt_handlers.append(p)
            plt_titles.append(label_titles[i])
        plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=0.5)
        result_file = osp.join(tempfile.mkdtemp(), 'result.png')
        plt.savefig(result_file, bbox_inches='tight', pad_inches=0)
        # compose result
        result_img = imread(result_file, mode='RGB')
        result_img = resize(result_img, img.shape, preserve_range=True)
        result_img = result_img.astype(np.uint8)
        # save result
        save_dir = osp.join(self.data_dir, 'forward_out')
        if not osp.exists(save_dir):
            save_dir = 'forward_out'
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
        print('- out_file: {0}'.format(out_file))


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
