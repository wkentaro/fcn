#!/usr/bin/env python

import argparse
import glob
import os.path as osp
import re
import subprocess
import tempfile

import scipy.misc

import fcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'log_dir', help='Log dir which contains log_XX.png and viz_XX.png')
    args = parser.parse_args()

    log_dir = args.log_dir

    tmpdir = tempfile.mkdtemp()

    for log_file in glob.glob(osp.join(log_dir, 'log_*.png')):
        # get log image
        img_log = scipy.misc.imread(log_file, mode='RGB')
        # get visualized image
        match = re.match('log_([0-9]*).png', osp.basename(log_file))
        iter_stop = int(match.groups()[0])
        viz_file = osp.join(log_dir, 'viz_{}.png'.format(iter_stop))
        img_viz = scipy.misc.imread(viz_file, mode='RGB')
        # save tiled image
        img_tiled = fcn.utils.get_tile_image(
            [img_log, img_viz], tile_shape=(2, 1),
            margin_color=(255, 255, 255))
        out_file = osp.join(tmpdir, '{}.png'.format(iter_stop))
        scipy.misc.imsave(out_file, img_tiled)

    # generate gif from images
    tmp_file = osp.join(tmpdir, '*.png')
    out_file = osp.join(log_dir, 'learning.gif')
    cmd = 'convert $(ls -v {}) gif:- | gifsicle -O3 --colors 256 > {}'\
        .format(tmp_file, out_file)
    subprocess.call(cmd, shell=True)
    print('wrote result: {}'.format(out_file))


if __name__ == '__main__':
    main()
