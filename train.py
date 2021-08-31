try:
    import colored_traceback
    colored_traceback.add_hook(always=True)
except:
    pass
import json
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.util import flatten, import_module
from tqdm import tqdm
from parse_config import ConfigParser
from collections import defaultdict
try:
    import nni
except:
    pass
try:
    import ax
    from ax import optimize
except:
    pass
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(config, args,  parameters=None, seed=None, pretrain=None):
    # -fix random seeds for reproducibility
    if 'seed' in config:
        SEED = config['seed']
    elif seed is not None:
        SEED = seed
    else:
        SEED = 123
    nni_dict = dict() # will hold final results

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # - ax/nni params
    if (parameters):
        config.mod_config(parameters)
    logger = config.get_logger('train')

    # - pretraining
    model = None
    pretrain = pretrain if pretrain is not None else args.pretrain
    if (pretrain):

        # - setup data_loader instances
        data_loader = config.init_obj('unsupervised_data_loader', module_data)
        valid_data_loader = data_loader.split_validation()
        # build model architecture, then print to console
        model = config.init_obj('unsupervised_arch', module_arch,
                    cat_idxs=getattr(data_loader.dataset, 'cat_idxs', []),
                    cat_dims=getattr(data_loader.dataset, 'cat_dims', []),
                    cont_idxs=getattr(data_loader.dataset, 'cont_idxs', []))
        logger.info(model)

        # - get function handles of loss and metrics
        criterion = None
        if hasattr(model, 'loss'):
            criterion = model.loss
        else:
            raise ValueError('provide loss function in config file/add loss implementation to model')

        metrics = [getattr(module_metric, met) for met in config['unsupervised_metrics']]
        # build optimizer, learning rate scheduler.
        # delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if 'unsupervised_optimizer_list' in config:
            optimizer = {}
            for opt, net in config['unsupervised_optimizer_list'].items():
                params = getattr(model, net).parameters()
                optimizer[opt] = config.init_obj(opt, torch.optim,
                        params)
        else:
            optimizer = config.init_obj('unsupervised_optimizer', torch.optim, trainable_params)

        # - setup optimizer/scheduler
        if type(optimizer) == dict:
            lr_scheduler = []
            for k, optim in optimizer.items():
                # note every optimizer/scheduler will use the same set of params..
                lr_scheduler.append(config.init_obj('unsupervised_lr_scheduler',
                    torch.optim.lr_scheduler, optim))
        else:
            lr_scheduler = config.init_obj('unsupervised_lr_scheduler',
                                       torch.optim.lr_scheduler, optimizer)

        # - import trainer
        Trainer = import_module('trainer', 'unsupervised_trainer', config)

        # - setup trainer
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          config_name='unsupervised_trainer',
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        result = trainer.train()
        if hasattr(trainer, 'pseudo_labeler'):
            nni_dict['pseudo_labeling_acc'] = trainer.data_loader.update_pseudo_labels(trainer.model, trainer.pseudo_labeler, trainer.device)

    if not pretrain or not len(data_loader.dataset.get_pseudo_labels()):
        data_loader = config.init_obj('supervised_data_loader', module_data)
        valid_data_loader = data_loader.split_validation()
    # train supervised model
    if model is not None:
        if 'unsupervisd_model_best' in config and config['unsupervised_model_best']:
            model = config.load_best_model(model)
        supervised_model = config.init_obj('supervised_arch', module_arch,
            encoder=model, input_dim=model.hidden_dim[-1])
    else:
        supervised_model = config.init_obj('supervised_arch', module_arch,
            cat_idxs=getattr(data_loader.dataset, 'cat_idxs', []),
            cat_dims=getattr(data_loader.dataset, 'cat_dims', []),
            cont_idxs=getattr(data_loader.dataset, 'cont_idxs', []),
            )

    logger.info(supervised_model)

    metrics = [getattr(module_metric, met) for met in config['supervised_metrics']]
    # build optimizer, learning rate scheduler.
    # delete every line containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, supervised_model.parameters())
    if 'supervised_optimizer_list' in config:
        optimizer = {}
        for opt, net in config['supervised_optimizer_list'].items():
            params = getattr(model, net).parameters()
            optimizer[opt] = config.init_obj(opt, torch.optim,
                    params)
    else:
        optimizer = config.init_obj('supervised_optimizer', torch.optim, trainable_params)

    if type(optimizer) == dict:
        lr_scheduler = []
        for k, optim in optimizer.items():
            # all will use the same scheduler params...
            lr_scheduler.append(config.init_obj('supervised_lr_scheduler',
                torch.optim.lr_scheduler, optim))
    else:
        lr_scheduler = config.init_obj('supervised_lr_scheduler',
                                   torch.optim.lr_scheduler, optimizer)

    if hasattr(supervised_model, 'loss'):
        criterion = supervised_model.loss
    else:
        raise ValueError('provide loss function in config file/add loss implementation to model')

    # - import supervised trainer
    Trainer = import_module('trainer', 'supervised_trainer', config)
    # - setup supervised trainer
    trainer = Trainer(supervised_model, criterion, metrics, optimizer,
                      config=config,
                      config_name='supervised_trainer',
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    result = trainer.train()
    # TODO: split checkpioint dirs for supervised/pretraining steps
    if 'supervised_model_best' in config and config['supervised_model_best']:
        supervised_model = config.load_best_model(supervised_model)
    # Test supervised model on test set and validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    supervised_model.eval()
    # Validation set
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    loss_fn = supervised_model.loss
    tape = defaultdict(list)
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_data_loader)):
            # data: list[Tensor] (last element is target)
            output = supervised_model(data, device, validation_target=False)
            tape['logits'].append(torch.nn.functional.softmax(output['logits']).cpu())
            tape['target'].append(data[-1].cpu())

            # computing loss, metrics on test set
            loss_dict = loss_fn(output, data, device)
            batch_size = data[-1].shape[0]
            loss = loss_dict['opt']
            total_loss += loss.item() * batch_size

    tape['logits'] = torch.cat(tape['logits'], dim=0)
    tape['target'] = torch.cat(tape['target'], dim=0)

    for i, metric in enumerate(metrics):
        total_metrics[i] = metric(tape['logits'].cpu(), tape['target'].long())
    n_samples = len(valid_data_loader.sampler)
    config_dict = flatten(config.config)
    config_dict['validation_accuracy'] = module_metric.accuracy(tape['logits'], tape['target'])

    for i, metric in enumerate(metrics):
        nni_dict['validation_' + metric.__name__] = total_metrics[i].item()
        config_dict['validation_' + metric.__name__] = total_metrics[i].item()

    # Test set
    test_dict = config['supervised_data_loader']['args']
    test_dict['batch_size'] = 512
    test_dict['shuffle'] = False
    test_dict['validation_split'] = 0.0
    test_dict['training'] = False
    test_dict['labeled_ratio'] = 0.0
    test_dict['num_workers'] = 2

    test_loader = getattr(module_data, config['supervised_data_loader']['type'])(
            **test_dict
        )
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    loss_fn = supervised_model.loss
    tape = defaultdict(list)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # data: list[Tensor] (last element is target)
            output = supervised_model(data, device, validation_target=False)
            tape['logits'].append(torch.nn.functional.softmax(output['logits']).cpu())
            tape['target'].append(data[-1].cpu())

            # computing loss, metrics on test set
            loss_dict = loss_fn(output, data, device)
            batch_size = data[-1].shape[0]
            loss = loss_dict['opt']
            total_loss += loss.item() * batch_size

    tape['logits'] = torch.cat(tape['logits'], dim=0)
    tape['target'] = torch.cat(tape['target'], dim=0)

    for i, metric in enumerate(metrics):
        total_metrics[i] = metric(tape['logits'].cpu(), tape['target'].long())
    n_samples = len(test_loader.sampler)
    config_dict = flatten(config.config)
    config_dict['test_accuracy'] = module_metric.accuracy(tape['logits'], tape['target'])

    for i, metric in enumerate(metrics):
        nni_dict['test_' + metric.__name__] = total_metrics[i].item()
        config_dict['test_' + metric.__name__] = total_metrics[i].item()

    nni_dict["default"] =  nni_dict['test_accuracy']
    nni.report_final_result(nni_dict)

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() for i, met in enumerate(metrics)
    })
    logger.info(log)
    # compute metrics
    #config_dict['roc_auc_macro'] = module_metric.roc_auc_score(tape['target'], tape['logits'], average='macro', multi_class='ovr')
    with open(config.log_dir / 'result.json', 'w') as f:
        json.dump(config_dict, f)

    # TODO: fix below
    #key = 'loss'
    #if (hasattr(trainer, 'mnt_metric')):
    #    key = trainer.mnt_metric

    #if key not in result:
    #    key = list(result.keys())[-1]
    #result = result[key]
    return nni_dict

def mod_config_nni(config, params):
    """
    params key, value to modify in ConfigParser object
    key should be in format
    <key_name_in_config>+<key_to_modify_in_config[key_name_in_config]>
    TODO: modify to set by path..
    """
    if params is None:
        return config
    for k, v in params.items():
        t = k.split('+')
        k2 = t[0]
        k3 = t[1]
        config.mod_key_config(k2, {k3: v})
    return config

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Tabular Data Augmentation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-rid', '--run_id', default=None, type=str,
                      help='run id to be used to save log/models within exp dir \
                      (default: current timestamp)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-ax', '--ax_config', default=None, type=str,
                      help='ax hyperparameter config file')
    args.add_argument('--pretrain', '-pretrain', default=False, action='store_true',
                      help='if pretrain is set encoder will be pretrained using configuration\
                            in config file')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    params = {}
    try:
        params = nni.get_next_parameter()
    except:
        pass
    config = mod_config_nni(config, params)
    args = args.parse_args()
    if (args.ax_config):
        parameters = json.loads(open(args.ax_config, 'r').read())
        best_parameters, values, experiment, model = optimize(
                parameters=parameters,
                evaluation_function=lambda p: main(config, args, p),
                minimize=True
                )
        print('*' * 30)
        print(best_parameters)
        print('*' * 30)
        ax.save(experiment, config.log_dir)
    else:
        main(config, args)
