#!/usr/bin/env python

import argparse
import copy
import json
import os
import os.path as osp

import chainer
from chainer import cuda
import numpy as np
import pandas as pd
import skimage.io
import skimage.util
import tqdm

import fcn

import datasets


def visualize_segmentation(lbl_pred, lbl_true, img, n_class):
    lbl_pred[lbl_true == -1] = 0
    lbl_true[lbl_true == -1] = 0

    cmap = fcn.util.labelcolormap(n_class)

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

    return fcn.util.get_tile_image(
        [viz_true0, viz_true1, viz_pred0, viz_pred1], tile_shape=(2, 2))


def write_log(log, out):
    log_file = osp.join(out, 'log.json')
    if osp.exists(log_file):
        logs = json.load(open(log_file))
    else:
        logs = []
    logs.append(log)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4, sort_keys=True)


def evaluate(iteration, model, iter_val, device, out, n_viz=9):
    model.train = False
    logs = []
    vizs = []
    dataset = iter_val.dataset
    interval = len(dataset) // n_viz
    desc = 'Evaluating at iteration %d' % iteration
    for batch in tqdm.tqdm(iter_val, desc=desc, total=len(dataset), ncols=80):
        in_arrays = [np.asarray(x) for x in zip(*batch)]
        if device >= 0:
            in_arrays = [cuda.to_gpu(x, device=device) for x in in_arrays]
        in_vars = [chainer.Variable(x) for x in in_arrays]
        model(*in_vars)
        logs.append(model.log)
        if iter_val.current_position % interval == 0 and len(vizs) < n_viz:
            img = dataset.datum_to_img(model.data[0])
            viz = visualize_segmentation(
                model.lbl_pred[0], model.lbl_true[0], img,
                n_class=model.n_class)
            vizs.append(viz)
    # # save visualization
    out_viz = osp.join(out, 'viz_eval', 'iter%d.jpg' % iteration)
    if not osp.exists(osp.dirname(out_viz)):
        os.makedirs(osp.dirname(out_viz))
    viz = fcn.utils.get_tile_image(vizs)
    skimage.io.imsave(out_viz, viz)
    # generate log
    log = pd.DataFrame(logs).mean(axis=0).to_dict()
    log = {'validation/%s' % k: v for k, v in log.items()}
    # finalize
    model.train = True
    return log


def train(model, optimizer, iter_train, iter_val, gpu,
          max_iter, interval_eval, out):
    for iteration, batch in enumerate(iter_train):

        # evaluate

        log_val = {}
        if iteration % interval_eval == 0:
            log_val = evaluate(iteration, model, copy.copy(iter_val),
                               device=gpu, out=out)
            out_model_dir = osp.join(out, 'models')
            if not osp.exists(out_model_dir):
                os.makedirs(out_model_dir)
            out_model = osp.join(out_model_dir, '%s_iter%d.h5' %
                                 (model.__class__.__name__, iteration))
            chainer.serializers.save_hdf5(out_model, model)

        # train

        in_arrays = [np.asarray(x) for x in zip(*batch)]
        if gpu >= 0:
            in_arrays = [cuda.to_gpu(x, device=gpu) for x in in_arrays]
        in_vars = [chainer.Variable(x) for x in in_arrays]

        model.zerograds()
        loss = model(*in_vars)

        if loss is not None:
            loss.backward()
            optimizer.update()
            log = model.log
            log['epoch'] = iter_train.epoch
            log['iteration'] = iteration
            log.update(log_val)
            write_log(model.log, out)

        if iteration >= max_iter:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fcn32s', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--dataset', default='v2', choices=['v1', 'v2'])
    args = parser.parse_args()

    fcn32s_path = args.fcn32s
    gpu = args.gpu
    out = args.out
    if args.dataset == 'v1':
        dataset_class = datasets.APC2016DatasetV1
    else:
        dataset_class = datasets.APC2016DatasetV2

    if not osp.exists(out):
        os.makedirs(out)

    # 1. dataset

    dataset_train = dataset_class('train')
    dataset_val = dataset_class('val')

    iter_train = chainer.iterators.SerialIterator(dataset_train, batch_size=1)
    iter_val = chainer.iterators.SerialIterator(dataset_val, batch_size=1,
                                                repeat=False, shuffle=False)

    # 2. model

    n_class = len(dataset_train.label_names)

    fcn32s = fcn.models.FCN32s(n_class=n_class)
    chainer.serializers.load_hdf5(fcn32s_path, fcn32s)

    model = fcn.models.FCN16s(n_class=n_class)
    model.train = True
    fcn.utils.copy_chainermodel(fcn32s, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=1e-5)
    optimizer.setup(model)

    # training loop

    train(
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_val=iter_val,
        gpu=gpu,
        max_iter=150000,
        interval_eval=5000,
        out=out,
    )


if __name__ == '__main__':
    main()
