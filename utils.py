import os
import numpy as np
import random
import torch as th
import logging
import sys
import ipdb
from torch.nn.modules.module import _addindent
from pprg_end2end.case_util import PAD, EOS, BOS
INT = 0
LONG = 1
FLOAT = 2

CUR_PATH = os.path.join(os.path.dirname(__file__))
DATA_PATH = CUR_PATH + '/../dataset'

class Pack(dict):  # an encapsulation class for dictionary
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return False

    def add(self, **kwargs):   # pass in dictionary
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(th.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(th.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.cuda.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    else:
        if dtype == INT:
            var = var.type(th.IntTensor)
        elif dtype == LONG:
            var = var.type(th.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    return var

def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def set_seed(seed):
    """Sets random seed everywhere."""
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if hasattr(config, 'forward_only') and config.forward_only:
        return

    fileHandler = logging.FileHandler(os.path.join(config.saved_path,'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    

def idx2word(vocab, data, b_id, stop_eos=True, stop_pad=True):
    """
        Translate indexs to real words(by vocabulary)
    """
    de_tknize = lambda x: ' '.join(x)
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w == BOS:
            continue
        if w != PAD:
            if w.startswith("##") and len(ws)>0:
                ws[-1] += w[2:]
            else:
                ws.append(w)
    return de_tknize(ws)

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from torch.utils.tensorboard import SummaryWriter

class TBLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.prepare_dir(log_dir) # clean previous tensorboard file first
        self.writer = SummaryWriter(log_dir)

    def prepare_dir(self, path):
        if os.path.exists(path):
            for f in os.listdir(path):
                file_path = os.path.join(path,f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(path)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, global_step=step)

    def add_scalar_summary(self, data_dict, step):
        for key,value in data_dict.items():
            self.scalar_summary(key, value, step)

def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            th.nn.modules.container.Container,
            th.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    tmpstr = tmpstr + '\n' + '-'*20
    return tmpstr