#!/usr/bin/env python

from __future__ import division

import argparse
import collections

import os.path as osp

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def learning_curve(json_file, output_interval):
    df = pd.read_json(json_file)

    colors = sns.husl_palette(3, l=.5, s=.5)

    plt.figure(figsize=(12, 6), dpi=500)

    # initialize DataFrame for train
    columns = [
        'iteration',
        'loss',
        'accuracy',
        'iu',
    ]
    df_train = df[columns]
    # get min/max
    row_max = df.max()
    # make smooth the learning curve with iteration step
    iteration_step = 10
    df_train_stat = []
    stat = collections.defaultdict(list)
    for index, row in df_train.iterrows():
        for col in row.keys():
            value = row[col]
            stat[col].append(value)
        if int(row['iteration']) % iteration_step == 0:
            means = [sum(stat[col]) / len(stat[col]) for col in row.keys()]
            means[0] = row['iteration']  # iteration_step is the representative
            df_train_stat.append(means)
            stat = collections.defaultdict(list)
    df_train = pd.DataFrame(df_train_stat, columns=df_train.columns)

    # initialize DataFrame for val
    columns = [
        'iteration',
        'validation/loss',
        'validation/accuracy',
        'validation/iu',
    ]
    df_val = df[columns]
    df_val = df_val.dropna()

    iter_start = 0
    if output_interval < 0:
        iter_start = int(row_max['iteration'])
        output_interval = 1

    for iter_stop in xrange(iter_start, int(row_max['iteration']) + 1,
                            output_interval):
        #########
        # TRAIN #
        #########

        df_train_step = df_train.query('iteration <= {}'.format(iter_stop))

        # train loss
        plt.subplot(231)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_train_step['iteration'], df_train_step['loss'], '-',
                 markersize=1, color=colors[0], alpha=.5, label='train loss')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['loss']))
        plt.xlabel('iteration')
        plt.ylabel('train loss')

        # train accuracy
        plt.subplot(232)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.plot(df_train_step['iteration'], df_train_step['accuracy'],
                 '-', markersize=1, color=colors[1], alpha=.5,
                 label='train accuracy')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['accuracy']))
        plt.xlabel('iteration')
        plt.ylabel('train overall accuracy')

        # train mean iu
        plt.subplot(233)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.plot(df_train_step['iteration'], df_train_step['iu'],
                 '-', markersize=1, color=colors[2], alpha=.5,
                 label='train accuracy')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['iu']))
        plt.xlabel('iteration')
        plt.ylabel('train mean IU')

        #######
        # VAL #
        #######

        df_val_step = df_val.query('iteration <= {}'.format(iter_stop))

        # val loss
        plt.subplot(234)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_val_step['iteration'], df_val_step['validation/loss'],
                 'o-', color=colors[0], alpha=.5, label='val loss')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['validation/loss']))
        plt.xlabel('iteration')
        plt.ylabel('val loss')

        # val accuracy
        plt.subplot(235)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.plot(df_val_step['iteration'],
                 df_val_step['validation/accuracy'],
                 'o-', color=colors[1], alpha=.5, label='val accuracy')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['validation/accuracy']))
        plt.xlabel('iteration')
        plt.ylabel('val overall accuracy')

        # val mean iu
        plt.subplot(236)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.plot(df_val_step['iteration'], df_val_step['validation/iu'],
                 'o-', color=colors[2], alpha=.5, label='val mean IU')
        plt.xlim((0, row_max['iteration']))
        plt.ylim((0, row_max['validation/iu']))
        plt.xlabel('iteration')
        plt.ylabel('val mean IU')

        fig_file = '{}_{}.png'.format(osp.splitext(json_file)[0], iter_stop)
        plt.savefig(fig_file)
        print('Saved to %s' % fig_file)

        plt.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('--output-interval', type=int, default=-1)
    args = parser.parse_args()

    json_file = args.json_file
    output_interval = args.output_interval

    learning_curve(json_file, output_interval)


if __name__ == '__main__':
    main()
