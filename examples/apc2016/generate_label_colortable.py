#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import fcn

from dataset import APC2016Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='2x20', help='default: 2x20')
    args = parser.parse_args()

    rows, cols = map(int, args.shape.split('x'))

    dataset = APC2016Dataset('val')
    n_lbl = len(dataset.label_names)
    table = fcn.utils.get_label_colortable(n_lbl, shape=(rows, cols))

    plt.imshow(table)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
