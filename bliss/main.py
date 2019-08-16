from __future__ import absolute_import
import argparse
import numpy as np
import torch
import sys
import os
import logging


from bliss.data import Language, Batcher
from bliss.models import Generator, Discriminator, GAN
from bliss.utils import setup_output_dir, read_from_yaml, disp_params


def get_arguments():
    def convert_to_boolean(args, name):
        if hasattr(args, name):
            assert getattr(args, name).lower() in set(["false", "true"]),\
                "Only boolean values allowed"
            val = True if getattr(args, name).lower() == "true" else False
            setattr(args, name, val)
        return args
    parser = argparse.ArgumentParser(
        description="Semi Supervised Word Embeddings")
    parser.add_argument('-cf', '--config_file',
                        action="store", dest="config_file",
                        type=str, help="path to the config file",
                        required=True)
    parser.add_argument('-gpu', '--gpu',
                        action="store", dest="gpu", type=str,
                        default="False", help="Use gpu",
                        required=False)
    parser.add_argument('-l', '--log', action="store",
                        dest="loglevel", type=str, default="DEBUG",
                        help="Logging Level")
    parser.add_argument('-s', '--seed', action="store",
                        dest="seed", type=int, default=-1,
                        help="use fixed random seed")
    parser.add_argument('-nt', '--num_threads', action="store",
                        dest="num_threads", type=int, default=4,
                        help="Fix number of cpu threads that FAISS can use")
    args = parser.parse_args(sys.argv[1:])
    args.loglevel = args.loglevel.upper()
    args = convert_to_boolean(args, 'gpu')
    args = convert_to_boolean(args, 'use_seed')
    return args


if __name__ == "__main__":
    params = get_arguments()
    config = read_from_yaml(params.config_file)
    if params.seed > 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.gpu:
            torch.cuda.manual_seed(params.seed)
    if params.gpu is False:
        torch.set_num_threads(params.num_threads)
    data_params = config['data_params']

    output_dir, config = setup_output_dir(
        data_params['output_dir'], config, params.loglevel)
    logger = logging.getLogger()
    disp_params(data_params, "data params")
    gen_params = config['generator_params']
    disp_params(gen_params, 'generator parameters')
    disc_params = config['discriminator_params']
    disp_params(disc_params, 'discriminator parameters')
    gan_params = config['gan_params']
    disp_params(gan_params, 'GAN parameters')
    train_params = config['train_params']
    disp_params(train_params, 'Training Parameters')
    generator = Generator(**gen_params)
    discriminator = Discriminator(**disc_params)
    languages = []
    # Load the data into languages
    data_dir = data_params['data_dir']
    for w in data_params['languages']:
        lang = Language(
            name=w['name'],
            gpu=params.gpu,
            mode=data_params['mode'],
            mean_center=data_params['mean_center'],
            unit_norm=data_params['unit_norm']
        )
        lang.load(w['filename'], data_dir, max_freq=75000)
        languages.append(lang)
    batcher = Batcher(languages)
    # Load the supervised data
    if 'supervised' in data_params:
        filename = data_params['supervised']['fname']
        freq = data_params['supervised']['max_freq']
        sup_dir_name = os.path.join(data_dir, "crosslingual", "dictionaries")
        batcher.load_from_supervised(
            filename, gan_params['src'], gan_params['tgt'],
            sup_dir_name, max_count=freq)
    if params.gpu:
        generator.cuda()
        discriminator.cuda()

    gan = GAN(
        discriminator,
        generator,
        batcher=batcher,
        gpu=params.gpu,
        data_dir=data_params['data_dir'],
        save_dir=output_dir, **gan_params)
    if 'supervised' in data_params and 'expand' in data_params['supervised'] and data_params['supervised']['expand']:
        # Expand dictionary once with procrustes
        batcher.expand_supervised(
            gan, gan_params['src'], gan_params['tgt'], train_params)

    if 'load_from' in data_params and data_params['load_from'] != '':
        assert os.path.exists(data_params['load_from'])
        gan.load_checkpoint(data_params['load_from'])

    if data_params['unsupervised']:
        if "supervised_method" in train_params and train_params["supervised_method"] == "rcsls":
            # Initialize with Procrustes for RCSLS
            src_str = gan_params["src"]
            tgt_str = gan_params["tgt"]
            word_dict = batcher.pair2ix[f"{src_str}-{tgt_str}"]
            pairs = word_dict.word_map
            weight = gan.procrustes_onestep(pairs)
            gan.gen.transform.weight.data.copy_(weight)
            logger.info("Initialized with Procrustes ...")
        gan.train(**train_params)
    else:
        logger = logging.getLogger()

        src_str = gan_params["src"]
        tgt_str = gan_params["tgt"]
        word_dict = batcher.pair2ix[f"{src_str}-{tgt_str}"]
        mode = data_params["supervised"]["mode"]
        if mode == "rcsls":
            params = {}
            params["mode"] = "csls"

            params["save"] = False
            params["niter"] = 1000
            params["batch_size"] = 32
            params["logafter"] = -1
            params["spectral"] = False
            params["k"] = 10
            params["num_tgts"] = 50000
            acc = gan.train_rcsls(
                word_dict.word_map, **params)
        else:
            acc = gan.train_artetxe(word_dict.word_map, 5, mode=mode)
