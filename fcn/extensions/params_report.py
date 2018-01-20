import os.path as osp
import pprint

from chainer import training
import yaml


def ParamsReport(params, file_name='params.yaml'):

    def initializer(trainer):
        print('# ' + '-' * 77)
        pprint.pprint(params)
        print('# ' + '-' * 77)
        with open(osp.join(trainer.out, file_name), 'w') as f:
            yaml.safe_dump(params, f, default_flow_style=False)

    @training.make_extension(initializer=initializer)
    def __call__(trainer):
        pass

    return __call__
