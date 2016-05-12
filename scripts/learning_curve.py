#!/usr/bin/env python

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def learning_curve(csv_file):
    df = pd.read_csv(csv_file)
    df_train  = df.query("type == 'train'")
    df_val = df.query("type == 'val'")

    plt.figure()

    # train loss
    plt.subplot(221)
    plt.semilogy(df_train.i_iter, df_train.loss, 'o', markersize=1, color='r',
                 alpha=.5, label='train loss')
    plt.title('train loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    # val loss
    plt.subplot(222)
    plt.semilogy(df_val.i_iter, df_val.loss, 'o-', color='r',
                 alpha=.5, label='val loss')
    plt.title('val loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    # train accuracy
    plt.subplot(223)
    plt.plot(df_train.i_iter, df_train.accuracy, 'o', markersize=1, color='g',
             alpha=.5, label='train accuracy')
    plt.title('train accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')

    # val accuracy
    plt.subplot(224)
    plt.plot(df_val.i_iter, df_val.accuracy, 'o-', color='g',
             alpha=.5, label='val accuracy')
    plt.title('val accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    args = parser.parse_args()
    learning_curve(args.csv_file)


if __name__ == '__main__':
    main()
