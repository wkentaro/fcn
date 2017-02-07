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
            iter_val,
            out,
            ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.iter_train = iter_train
        self.iter_val = iter_val
        self.out = out
        self.epoch = 0
        self.iteration = 0

    def evaluate(self, n_viz=9):
        iter_val = copy.copy(self.iter_val)
        self.model.train = False
        logs = []
        vizs = []
        dataset = iter_val.dataset
        interval = len(dataset) // n_viz
        desc = 'eval [iter=%d]' % self.iteration
        for batch in tqdm.tqdm(iter_val, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            in_vars = utils.batch_to_vars(
                batch, device=self.device, volatile=True)
            self.model(*in_vars)
            logs.append(self.model.log)
            if iter_val.current_position % interval == 0 and \
                    len(vizs) < n_viz:
                img = dataset.datum_to_img(self.model.data[0])
                viz = utils.visualize_segmentation(
                    self.model.lbl_pred[0], self.model.lbl_true[0], img,
                    n_class=self.model.n_class)
                vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'viz_eval', 'iter%d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = fcn.utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        log = pd.DataFrame(logs).mean(axis=0).to_dict()
        log = {'validation/%s' % k: v for k, v in log.items()}
        # finalize
        self.model.train = True
        return log

    def train(self, max_iter, interval_eval):
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # evaluate #
            ############

            log_val = {}
            if iteration % interval_eval == 0:
                log_val = self.evaluate()
                out_model_dir = osp.join(self.out, 'models')
                if not osp.exists(out_model_dir):
                    os.makedirs(out_model_dir)
                out_model = osp.join(
                    out_model_dir, '%s_iter%d.h5' %
                    (self.model.__class__.__name__, self.iteration))
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
                log = self.model.log
                log['epoch'] = self.iter_train.epoch
                log['iteration'] = iteration
                log.update(log_val)
                utils.append_log_to_json(self.model.log,
                                         osp.join(self.out, 'log.json'))

            if iteration >= max_iter:
                break
