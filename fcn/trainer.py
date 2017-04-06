import collections
import copy
import os
import os.path as osp

import chainer
import pandas as pd
import skimage.io
import skimage.util
import tqdm

import fcn
from fcn import utils


class Trainer(object):

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.iter_train = iter_train
        self.iter_valid = iter_valid
        self.out = out
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def evaluate(self, n_viz=9):
        iter_valid = copy.copy(self.iter_valid)
        self.model.train = False
        logs = []
        vizs = []
        dataset = iter_valid.dataset
        interval = len(dataset) // n_viz
        desc = 'eval [epoch=%d]' % self.epoch
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            in_vars = utils.batch_to_vars(
                batch, device=self.device, volatile=True)
            self.model(*in_vars)
            logs.append(self.model.log)
            if iter_valid.current_position % interval == 0 and \
                    len(vizs) < n_viz:
                img = dataset.datum_to_img(self.model.data[0])
                viz = utils.visualize_segmentation(
                    self.model.lbl_pred[0], self.model.lbl_true[0], img,
                    n_class=self.model.n_class)
                vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'viz_eval', 'epoch%d.jpg' % self.epoch)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        log = pd.DataFrame(logs).mean(axis=0).to_dict()
        log = {'valid/%s' % k: v for k, v in log.items()}
        # finalize
        self.model.train = True
        return log

    def train(self):
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # evaluate #
            ############

            if self.iteration == 0 or self.iter_train.is_new_epoch:
                log = collections.defaultdict(str)
                log_valid = self.evaluate()
                log.update(log_valid)
                log['epoch'] = self.iter_train.epoch
                log['iteration'] = iteration
                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    f.write(','.join(str(log[h]) for h in self.log_headers) +
                            '\n')
                out_model_dir = osp.join(self.out, 'models')
                if not osp.exists(out_model_dir):
                    os.makedirs(out_model_dir)
                out_model = osp.join(
                    out_model_dir, '%s_epoch%d.h5' %
                    (self.model.__class__.__name__, self.epoch))
                chainer.serializers.save_hdf5(out_model, self.model)

            #########
            # train #
            #########

            in_vars = utils.batch_to_vars(
                batch, device=self.device, volatile=False)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()
                log = collections.defaultdict(str)
                log_train = {'train/%s' % k: v
                             for k, v in self.model.log.items()}
                log['epoch'] = self.iter_train.epoch
                log['iteration'] = iteration
                log.update(log_train)
                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    f.write(','.join(str(log[h]) for h in self.log_headers) +
                            '\n')

            if iteration >= self.max_iter:
                break
