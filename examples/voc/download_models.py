#!/usr/bin/env python

import fcn


def main():
    # models converted from caffe
    path = fcn.data.download_vgg16_chainermodel()
    print('==> downloaded to: %s' % path)
    path = fcn.data.download_fcn32s_chainermodel()
    print('==> downloaded to: %s' % path)
    path = fcn.data.download_fcn16s_chainermodel()
    print('==> downloaded to: %s' % path)
    path = fcn.data.download_fcn8s_chainermodel()
    print('==> downloaded to: %s' % path)


if __name__ == '__main__':
    main()
