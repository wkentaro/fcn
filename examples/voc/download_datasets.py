#!/usr/bin/env python

import fcn


def main():
    fcn.datasets.VOC2012ClassSeg.download()
    fcn.datasets.SBDClassSeg.download()


if __name__ == '__main__':
    main()
