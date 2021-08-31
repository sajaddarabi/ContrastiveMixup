import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
import collections

import torch

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, **kwargs):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['supervised_trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S.%f') + kwargs.get('run_id_post_fix', '')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.

        if (kwargs.get('mkdirs', True)):
            exist_ok = run_id == ''
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

            # save updated config file to the checkpoint dir
            write_json(self.config, self.save_dir / 'config.json')

            # configure logging module
            setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options='', **kwargs):
        """
        initialize this class from some cli arguments. used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["cuda_visible_devices"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "configuration file need to be specified. add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)

        if hasattr(args, 'config') and args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        if 'run_id' in args and 'run_id' not in kwargs:
            kwargs = { **kwargs, 'run_id': args.run_id }
        return cls(config, resume, modification, **kwargs)

    def write_config(self, save_dir):
            write_json(self.config, save_dir / 'config.json')

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        #assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def __contains__(self, key):
        return key in self._config

    def mod_config(self, params):
        for k, v in params.items():
            if k in self._config:
                self._config[k] = v
                continue
            for kk, vv in self._config.items():
                if (type(vv) == collections.OrderedDict):
                    self._config[kk] = recurse_dict(vv, k, v)
        # write modified config
        write_json(self.config, self.save_dir / 'config.json')

    def mod_key_config(self, key, params):
        assert key in self._config, f'{key} is not present in config file'
        for k, v in params.items():
            if k in self._config[key]:
                self._config[key][k] = v
                continue
            for kk, vv in self._config[key].items():
                if (type(vv) == collections.OrderedDict):
                    self._config[key][kk] = recurse_dict(vv, k, v)
        write_json(self.config, self.save_dir / 'config.json')

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def load_best_model(self, model):
        resume_path = self.save_dir / 'model_best.pth'
        logger = self.get_logger('model', 1)
        if not resume_path.exists():
            logger.info("Could not find checkpoint: {} ...".format(resume_path))
            return model
        logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

# TODO: clean using _update_config?
def recurse_dict(d, k, v):
        if (k in d):
            d[k] = v
            return d
        for kk, vv in d.items():
            if (type(vv) == collections.OrderedDict):
                d[kk] = recurse_dict(vv, k, v)
        return d

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
