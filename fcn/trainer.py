import collections
import copy
import os
import os.path as osp
import time

import chainer
import numpy as np
import skimage.io
import skimage.util
import tqdm

from . import datasets
from . import utils


class Trainer(object):

    """Training class for FCN models.

    Parameters
    ----------
    device: int
        GPU id, negative values represents use of CPU.
    model: chainer.Chain
        NN model.
    optimizer: chainer.Optimizer
        Optimizer.
    iter_train: chainer.Iterator
        Dataset itarator for training dataset.
    iter_valid: chainer.Iterator
        Dataset itarator for validation dataset.
    out: str
        Log output directory.
    max_iter: int
        Max iteration to stop training iterations.
    interval_validate: None or int
        If None, validation is never run. (default: 4000)

    Returns
    -------
    None
    """

    def __init__(
            self,
            device,
            model,
            optimizer,
            iter_train,
            iter_valid,
            out,
            max_iter,
            interval_validate=4000):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.iter_train = iter_train
        self.iter_valid = iter_valid
        self.out = out
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.interval_validate = interval_validate
        self.stamp_start = None
        # for logging
        self.log_headers = [
            'epoch',
            'iteration',
            'elapsed_time',
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
        if not osp.exists(self.out):
            os.makedirs(self.out)
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def validate(self, n_viz=9):
        """Validate current model using validation dataset.

        Parameters
        ----------
        n_viz: int
            Number fo visualization.

        Returns
        -------
        log: dict
            Log values.
        """
        iter_valid = copy.copy(self.iter_valid)
        losses, lbl_trues, lbl_preds = [], [], []
        vizs = []
        dataset = iter_valid.dataset
        desc = 'valid [iteration=%08d]' % self.iteration
        for batch in tqdm.tqdm(iter_valid, desc=desc, total=len(dataset),
                               ncols=80, leave=False):
            img, lbl_true = zip(*batch)
            batch = map(datasets.transform_lsvrc2012_vgg16, batch)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                in_vars = utils.batch_to_vars(batch, device=self.device)
                loss = self.model(*in_vars)
            losses.append(float(loss.data))
            score = self.model.score
            lbl_pred = chainer.functions.argmax(score, axis=1)
            lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
            for im, lt, lp in zip(img, lbl_true, lbl_pred):
                lbl_trues.append(lt)
                lbl_preds.append(lp)
                if len(vizs) < n_viz:
                    viz = utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt,
                        img=im, n_class=self.model.n_class)
                    vizs.append(viz)
        # save visualization
        out_viz = osp.join(self.out, 'visualizations_valid',
                           'iter%08d.jpg' % self.iteration)
        if not osp.exists(osp.dirname(out_viz)):
            os.makedirs(osp.dirname(out_viz))
        viz = utils.get_tile_image(vizs)
        skimage.io.imsave(out_viz, viz)
        # generate log
        acc = utils.label_accuracy_score(
            lbl_trues, lbl_preds, self.model.n_class)
        self._write_log(**{
            'epoch': self.epoch,
            'iteration': self.iteration,
            'elapsed_time': time.time() - self.stamp_start,
            'valid/loss': np.mean(losses),
            'valid/acc': acc[0],
            'valid/acc_cls': acc[1],
            'valid/mean_iu': acc[2],
            'valid/fwavacc': acc[3],
        })
        self._save_model()

    def _write_log(self, **kwargs):
        log = collections.defaultdict(str)
        log.update(kwargs)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            f.write(','.join(str(log[h]) for h in self.log_headers) + '\n')

    def _save_model(self):
        out_model_dir = osp.join(self.out, 'models')
        if not osp.exists(out_model_dir):
            os.makedirs(out_model_dir)
        model_name = self.model.__class__.__name__
        out_model = osp.join(out_model_dir, '%s_iter%08d.npz' %
                             (model_name, self.iteration))
        chainer.serializers.save_npz(out_model, self.model)

    def train(self):
        """Train the network using the training dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stamp_start = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(self.iter_train),
                                          desc='train', total=self.max_iter,
                                          ncols=80):
            self.epoch = self.iter_train.epoch
            self.iteration = iteration

            ############
            # validate #
            ############

            if self.interval_validate and \
                    self.iteration % self.interval_validate == 0:
                self.validate()

            #########
            # train #
            #########

            batch = map(datasets.transform_lsvrc2012_vgg16, batch)
            in_vars = utils.batch_to_vars(batch, device=self.device)
            self.model.zerograds()
            loss = self.model(*in_vars)

            if loss is not None:
                loss.backward()
                self.optimizer.update()

                lbl_true = zip(*batch)[1]
                lbl_pred = chainer.functions.argmax(self.model.score, axis=1)
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)
                acc = utils.label_accuracy_score(
                    lbl_true, lbl_pred, self.model.n_class)
                self._write_log(**{
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'elapsed_time': time.time() - self.stamp_start,
                    'train/loss': float(loss.data),
                    'train/acc': acc[0],
                    'train/acc_cls': acc[1],
                    'train/mean_iu': acc[2],
                    'train/fwavacc': acc[3],
                })

            if iteration >= self.max_iter:
                self._save_model()
                break
