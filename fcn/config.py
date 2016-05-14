from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp


def _get_data_dir():
    this_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.realpath(osp.join(this_dir, '_data'))
    if osp.exists(data_dir):
        return data_dir
    return ''
