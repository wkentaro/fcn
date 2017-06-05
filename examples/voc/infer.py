#!/usr/bin/env python

from __future__ import division

import argparse
import os.path as osp

import chainer
import scipy.misc

import fcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-c', '--chainermodel')
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    args = parser.parse_args()

    img_files = args.img_files
    gpu = args.gpu
    if args.chainermodel is None:
        chainermodel = fcn.data.download_fcn8s_chainermodel()
    else:
        chainermodel = args.chainermodel
    save_dir = chainer.dataset.get_dataset_directory('fcn/inference')

    dataset = fcn.datasets.PascalVOC2012SegmentationDataset('val')

    if osp.basename(chainermodel).lower().startswith('fcn32s'):
        model_class = fcn.models.FCN32s
    elif osp.basename(chainermodel).lower().startswith('fcn16s'):
        model_class = fcn.models.FCN16s
    elif osp.basename(chainermodel).lower().startswith('fcn8s'):
        model_class = fcn.models.FCN8s
    else:
        raise ValueError
    model = model_class(n_class=len(dataset.label_names))
    chainer.serializers.load_npz(chainermodel, model)

    infer = fcn.Inferencer(dataset, model, gpu)
    for img_file in img_files:
        img, label = infer.infer_image_file(img_file)
        out_img = infer.visualize_label(img, label)

        out_file = osp.join(save_dir, osp.basename(img_file))
        scipy.misc.imsave(out_file, out_img)
        print('- out_file: {0}'.format(out_file))


if __name__ == '__main__':
    main()
