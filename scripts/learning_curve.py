#!/usr/bin/env python

from __future__ import division

import argparse
import collections

import os.path as osp

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def learning_curve(json_file):
    df = pd.read_json(json_file)

    colors = sns.husl_palette(3, l=.5, s=.5)

    plt.figure(figsize=(12, 6), dpi=500)

    #########
    # TRAIN #
    #########

    columns = [
        'iteration',
        'main/loss',
        'main/accuracy',
        'main/iu',
    ]
    df_train = df[columns]

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

    # train loss
    plt.subplot(231)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.plot(df_train['iteration'], df_train['main/loss'],
             '-', markersize=1, color=colors[0], alpha=.5, label='train loss')
    plt.xlabel('iteration')
    plt.ylabel('train loss')

    # train accuracy
    plt.subplot(232)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_train['iteration'], df_train['main/accuracy'],
             '-', markersize=1, color=colors[1], alpha=.5,
             label='train accuracy')
    plt.xlabel('iteration')
    plt.ylabel('train overall accuracy')

    # train mean iu
    plt.subplot(233)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_train['iteration'], df_train['main/iu'],
             '-', markersize=1, color=colors[2], alpha=.5,
             label='train accuracy')
    plt.xlabel('iteration')
    plt.ylabel('train mean IU')

    #######
    # VAL #
    #######

    columns = [
        'iteration',
        'validation/main/loss',
        'validation/main/accuracy',
        'validation/main/iu',
    ]
    df_val = df[columns]
    df_val = df_val.dropna()

    # val loss
    plt.subplot(234)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(df_val['iteration'], df_val['validation/main/loss'],
             'o-', color=colors[0], alpha=.5, label='val loss')
    plt.xlabel('iteration')
    plt.ylabel('val loss')

    # val accuracy
    plt.subplot(235)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_val['iteration'], df_val['validation/main/accuracy'],
             'o-', color=colors[1], alpha=.5, label='val accuracy')
    plt.xlabel('iteration')
    plt.ylabel('val overall accuracy')

    # val mean iu
    plt.subplot(236)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.plot(df_val['iteration'], df_val['validation/main/iu'],
             'o-', color=colors[2], alpha=.5, label='val mean IU')
    plt.xlabel('iteration')
    plt.ylabel('val mean IU')

    fig_file = osp.splitext(json_file)[0] + '.png'
    plt.savefig(fig_file)
    print('Saved to %s' % fig_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    learning_curve(args.json_file)


if __name__ == '__main__':
    main()
