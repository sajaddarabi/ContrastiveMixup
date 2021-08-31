import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from collections import defaultdict
from model.pseudo_labelers import GraphLabelPropagation

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, config_name, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config_name, config)
        self.config = config
        self.config_name = config_name
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.validation_target = config[config_name].get('validation_target', False)
        self.log_step = config[config_name].get('log_step', int(np.sqrt(data_loader.batch_size)))
        self.val_loss_to_track = config[config_name].get('val_loss_to_track', 'opt')
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.config[self.config_name].get('pseudolabeling', None) and \
           self.config[self.config_name]['pseudolabeling'].get('epoch_start') >= 0:
            if not hasattr(self, 'pseudolabeler'):
                self.pseudo_labeler = GraphLabelPropagation(**self.config[self.config_name]['pseudolabeling']['args'])
            epoch_start = self.config[self.config_name]['pseudolabeling']['epoch_start']
            every_f_epoch = self.config[self.config_name]['pseudolabeling']['f']
            if (epoch > epoch_start) and \
               (((epoch - epoch_start) % every_f_epoch) == 0):
                   acc = self.data_loader.update_pseudo_labels(self.model, self.pseudo_labeler, self.device)
                   self.logger.debug('Train Epoch: {} PL-Accuracy: {:.6f}'.format(
                        epoch,
                        acc * 100,
                        ))


        self.model.train()
        self.train_metrics.reset()
        in_out_tape = defaultdict(list)
        loss_tape = defaultdict(list)
        batch_lengths = list()
        total_loss = 0.0
        for batch_idx, data in enumerate(self.data_loader):
            x, x_u, target, utarget, weight = None, None, None, None, None
            if type(data) == torch.Tensor:
                x = data
            elif len(data) == 3:
                x, x_u, target = data
            elif len(data) == 4:
                import pdb; pdb.set_trace()
                x, x_u, target, utarget = data
            elif len(data) == 5:
                x, x_u, target, utarget, _ = data
            else:
                x, target = data

            self.optimizer.zero_grad()
            output = self.model(data, self.device)
            loss_args = {} if 'loss_args' not in self.config else self.config['loss_args']
            loss = self.criterion(output, data, self.device, **loss_args)

            # - debug info
            debug_str = ''
            if (type(loss) == dict):
                loss_dict = loss
                for k, v in loss_dict.items():
                    v = v if type(v) != torch.Tensor else v.item()
                    loss_tape[k].append(v) # record loss info
                    debug_str += '{}: {:.6f} '.format(k, v)
                debug_str = debug_str.strip()
                loss = loss['opt']
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(data[-1])

            # - tape
            in_out_tape['preds'].append(output['preds'].detach().cpu())
            in_out_tape['target'].append(target)
            batch_lengths.append(len(data[-1]))

            # - logging
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(), debug_str))

            if batch_idx == self.len_epoch:
                break

        in_out_tape['preds'] = torch.cat(in_out_tape['preds'], dim=0).detach()
        in_out_tape['target'] = torch.cat(in_out_tape['target'], dim=0).detach()
        #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'train')

        total_examples = sum(batch_lengths)
        self.writer.set_step((epoch - 1), 'train')
        self.train_metrics.update('loss', total_loss / total_examples)

        # classification metrics
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(in_out_tape['preds'], in_out_tape['target'].long()))

        loss_tape_results = {k: sum(map(lambda x: x[0] * x[1], list(zip(v, batch_lengths)))) / total_examples \
                for k, v in loss_tape.items()}

        self.writer.set_step((epoch - 1), 'train')
        for k, v in loss_tape_results.items():
            self.writer.add_scalar(k, v)


        if (hasattr(self.model, 'training_step_cb')):
            self.model.training_step_cb(epoch)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        in_out_tape = defaultdict(list)
        loss_tape = defaultdict(list)
        batch_lengths = list()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                # - inputs
                x, x_u, target, utarget, weight = None, None, None, None, None
                if type(data) == torch.Tensor:
                    x = data
                elif len(data) == 3:
                    x, x_u, target = data
                elif len(data) == 4:
                    x, x_u, target, utarget = data
                elif len(data) == 5:
                    x, x_u, target, utarget, _ = data
                else:
                    x, target = data

                output = self.model(data, self.device,
                                    validation_target=self.validation_target)

                in_out_tape['preds'].append(output['preds'].cpu())
                in_out_tape['target'].append(target)
                loss_args = {} if 'loss_args' not in self.config else self.config['loss_args']
                loss = self.criterion(output, data, self.device, **loss_args)

                debug_str = ''
                if (type(loss) == dict):
                    loss_dict = loss
                    for k, v in loss_dict.items():
                        v = v if type(v) != torch.Tensor else v.item()
                        loss_tape[k].append(v)
                        debug_str += '{}: {:.6f} '.format(k, v)
                    loss = loss_dict[self.val_loss_to_track]
                    debug_str = debug_str.strip()
                else:
                    loss_tape['loss'].append(v.item())

                total_loss += loss.item() * len(data[-1])

                batch_lengths.append(len(data[-1]))

        in_out_tape['preds'] = torch.cat(in_out_tape['preds'], dim=0).detach()
        in_out_tape['target'] = torch.cat(in_out_tape['target'], dim=0).detach()
        total_examples = sum(batch_lengths)
        self.writer.set_step((epoch - 1), 'valid')
        self.valid_metrics.update('loss', total_loss / total_examples)
        # classification metrics
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, met(in_out_tape['preds'], in_out_tape['target'].long()))

        loss_tape_results = {k: sum(map(lambda x: x[0] * x[1], list(zip(v, batch_lengths)))) / total_examples \
                for k, v in loss_tape.items()}

        self.writer.set_step((epoch - 1), 'valid')
        for k, v in loss_tape_results.items():
            self.writer.add_scalar(k, v)
        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
