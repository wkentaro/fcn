#!/usr/bin/env python

import fcn


def main():
    fcn.models.VGG16.download()
    fcn.models.FCN32s.download()
    fcn.models.FCN16s.download()
    fcn.models.FCN8s.download()
    fcn.models.FCN8sAtOnce.download()


if __name__ == '__main__':
    main()
