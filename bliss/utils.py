import numpy as np
import os
import sys
import yaml
import logging
from subprocess import Popen, PIPE
import torch.autograd
import torch.nn


def print_metrics(metrics, fp=None):
    metric_str = ""
    for metric in metrics:
        metric_str += '\t%s: %.4f' % (metric, metrics[metric])
    if fp is None:
        print(metric_str)
    else:
        with open(fp, 'wb') as f:
            f.write(metric_str)


def setup_logger(loglevel, logfile=None):
    """Sets up the logger

    Arguments:
        loglevel (str): The log level (INFO|DEBUG|..)
        logfile Optional[str]: Add a file handle

    Returns:
        None
    """
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=numeric_level, stream=sys.stdout)
    if logfile is not None:
        fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
        logfile_handle = logging.FileHandler(logfile, 'w')
        logfile_handle.setFormatter(fmt)
        logger.addHandler(logfile_handle)


def setup_output_dir(output_dir, config, loglevel):
    """
        Takes in the output_dir. Note that the output_dir stores each run as run-1, ....
        Makes the next run directory. This also sets up the logger
        A run directory has the following structure
        run-1:
            |_ best_model
                     |_ model_params_and_metrics.tar.gz
                     |_ validation paths.txt
            |_ last_model_params_and_metrics.tar.gz
            |_ config.yaml
            |_ githash.log of current run
            |_ gitdiff.log of current run
            |_ logfile.log (the log of the current run)
        This also changes the config, to add the save directory
    """
    make_directory(output_dir, recursive=True)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith('run-'):
            last_run = max(last_run, int(dirname.split('-')[1]))
    new_dirname = os.path.join(output_dir, 'run-%d' % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, 'best_model')
    make_directory(best_model_dirname)
    config_file = os.path.join(new_dirname, 'config.yaml')
    config['data_params']['save_dir'] = new_dirname
    write_to_yaml(config_file, config)
    # Save the git hash
    process = Popen('git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8').strip('\n').strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)
    # Save the git diff
    process = Popen('git diff'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode('utf-8')
        fp.write(stdout)
    # Set up the logger
    logfile = os.path.join(new_dirname, 'logfile.log')
    setup_logger(loglevel, logfile)
    return new_dirname, config


def read_from_yaml(filepath):
    with open(filepath, 'r') as fd:
        data = yaml.load(fd)
    return data


def write_to_yaml(filepath, data):
    with open(filepath, 'w') as fd:
        yaml.dump(data=data, stream=fd, default_flow_style=False)


def make_directory(dirname, recursive=False):
    os.makedirs(dirname, exist_ok=True)


def disp_params(params, name):
        print_string = "{0}".format(name)
        for param in params:
            print_string += '\n\t%s: %s' % (param, str(params[param]))
        logger = logging.getLogger()
        logger.info(print_string)


def to_cuda(t, gpu):
    return t.cuda() if gpu else t


def to_numpy(t, gpu):
    """
        Takes in a Variable, and returns numpy
    """
    ret = t.data if isinstance(t, (torch.autograd.Variable, torch.nn.Parameter)) else t
    ret = ret.cpu() if gpu else ret  # this brings it back to cpu
    return ret.numpy()
