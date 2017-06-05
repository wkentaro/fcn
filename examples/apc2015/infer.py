#!/usr/bin/env python

import os.path as osp
import sys


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, osp.join(here, '../voc'))

    import infer
    infer.infer(n_class=26)
