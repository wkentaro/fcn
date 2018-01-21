import os.path as osp
import pprint

from chainer import training
import yaml


class ParamsReport(training.Extension):

    def __init__(self, params, file_name='params.yaml'):
        self._params = params
        self._file_name = file_name

    def initialize(self, trainer):
        print('# ' + '-' * 77)
        pprint.pprint(self._params)
        print('# ' + '-' * 77)
        with open(osp.join(trainer.out, self._file_name), 'w') as f:
            yaml.safe_dump(self._params, f, default_flow_style=False)
