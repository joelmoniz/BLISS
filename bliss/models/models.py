from __future__ import absolute_import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import logging
import torch.nn.functional as F
import os
import numpy as np
from copy import deepcopy
import time
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bliss.utils import to_cuda, to_numpy
from bliss.eval import CSLS
from bliss.data import WordDictionary, MonoDictionary, Language,\
    CrossLingualDictionary, GaussianAdditive


class Discriminator(nn.Module):

    def __init__(self, embed_dim, hidden_dim=2048,
                 dropout_prob=0.1, max_freq=-1):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, inp):
        return self.mlp(inp)

    def init_weights(self):
        def init(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight.data)
        self.apply(init)


class Generator(nn.Module):
    def __init__(self, embed_dim, init):
        super(Generator, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim, bias=False)
        self.init_weights(init)
        self.init = init

    def forward(self, inp):
        return self.transform(inp)

    def _initialize(self):
        assert hasattr(self, 'init')
        self.init_weights(self.init)

    def init_weights(self, init):
        if init == 'ortho':
            nn.init.orthogonal(
                self.transform.weight, gain=nn.init.calculate_gain('linear'))
        elif init == 'eye':
            nn.init.eye_(self.transform.weight)
        else:
            raise NotImplementedError("{0} not supported".format(init))

    def orthogonalize(self, ortho_type="basic", beta=0.001):
        """
        Perform the orthogonalization step on the generated W matrix
        """
        if ortho_type == "basic":
            W = self.transform.weight.data
            o = ((1 + beta) * W) - (beta * W.mm(W.t().mm(W)))
            W.copy_(o)
        elif ortho_type == "spectral":
            self.spectral()
        elif ortho_type == "forbenius":
            self.forbenius()
        else:
            raise NotImplementedError(f"{ortho_type} not found")

    def spectral(self):
        W = self.transform.weight.data
        u, sigma, v = torch.svd(W)
        sigma_clamped = sigma.clamp(max=1.)
        new_W = torch.mm((torch.mm(u, torch.diag(sigma_clamped))), v.t())
        W.copy_(new_W)

    def forbenius(self):
        W = self.transform.weight.data
        fnorm = (W ** 2).sum() + 1e-6
        new_W = W / fnorm
        W.copy_(new_W)


class MonitorLR(object):
    def __init__(self, name, optimizer, min_lr, factor,
                 shrink=None, patience=None):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.factor = factor
        self.logger = logging.getLogger()
        self.name = name
        self.best_metric = None
        self.bad_runs = -1
        self.shrink = shrink
        self.patience = patience

    def step(self, metric=None):
        old_lr = set()
        new_lr = set()
        for param_group in self.optimizer.param_groups:
            old_lr.add(param_group['lr'])
            param_group['lr'] = max(
                self.min_lr, param_group['lr'] * self.factor)
            new_lr.add(param_group['lr'])
        assert len(old_lr) == len(new_lr) and len(old_lr) == 1,\
            "Multi lr not supported"
        old_lr = list(old_lr)[0]
        new_lr = list(new_lr)[0]
        if old_lr != new_lr:
            self.logger.info(
                "Changing {0:s} Learning Rate: {1:.7f} -> {2:.7f}".format(
                    self.name, old_lr, new_lr))
        if self.shrink is not None:
            if self.best_metric is None or self.best_metric < metric:
                self.best_metric = metric
                self.bad_runs = 0
                return
            self.bad_runs += 1
            if self.bad_runs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(
                        self.min_lr, param_group['lr'] * self.shrink)
                shrunk_lr = new_lr * self.shrink
                bad_runs = self.bad_runs
                self.bad_runs = 0
                self.logger.info(
                    f"Shrinking LR: {new_lr:.7f} -> {shrunk_lr:.7f}"
                    f" After {bad_runs:d} Bad runs")


class Evaluator(object):
    """
        Computes different metrics for evaluation purposes.
        Currently supported evaluation metrics
            - Unsupervised CSLS metric for model selection
            - Cross-Lingual precision matches
    """

    def __init__(self, src_lang, tgt_lang, data_dir, save_dir):
        # Load Cross-Lingual Data
        """
            :param src_lang (Language): The source Language
            :param tgt_lang (Language): The Target Language
            :param data_dir (str): The data directory
                (assumes cross-lingual dictionaries are kept in
                data_dir/crosslingual/{src}-{tgt}.txt)
        """
        assert (isinstance(src_lang, Language) and
                isinstance(tgt_lang, Language) and
                isinstance(data_dir, str))
        self.data_dir = data_dir
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.logger = logging.getLogger()
        self.save_dir = save_dir

    def supervised(
        self, csls, metrics,
        mode="csls", nltk_flag=False, word_dict=None
    ):
        """
        Reports the precision at k (1, 5, 10) accuracy for
        word translation using facebook dictionaries
        """
        try:
            import nltk
            nltk.data.path = ['./nltk_data']
        except ImportError:
            pass
        if not hasattr(self, 'word_dict'):
            cross_lingual_dir = os.path.join(
                self.data_dir, "crosslingual", "dictionaries")
            cross_lingual_file = "%s-%s.5000-6500.txt" % (
                self.src_lang.name, self.tgt_lang.name)
            cross_lingual_file = os.path.join(
                cross_lingual_dir, cross_lingual_file)
            self.word_dict = WordDictionary(
                self.src_lang, self.tgt_lang, cross_lingual_file)
        word_dict = word_dict or self.word_dict
        predictions, _ = self.get_match_samples(
            csls, word_dict.word_map[:, 0], 10, mode=mode, use_mean=False)
        _metrics, total = word_dict.precisionatk_nltk(
            predictions, [1, 5, 10]) \
            if nltk_flag else \
            word_dict.precisionatk(predictions, [1, 5, 10])
        metrics['total'] = total
        metrics['acc1'] = _metrics[0]
        metrics['acc5'] = _metrics[1]
        metrics['acc10'] = _metrics[2]
        total = metrics["total"]
        acc1 = metrics["acc1"]
        acc5 = metrics["acc5"]
        acc10 = metrics["acc10"]
        self.logger.info(
            f"Total: {total:d}, "
            f"Precision@1: {acc1:5.2f}, @5: {acc5:5.2f}, @10: {acc10:5.2f}"
        )
        return metrics

    def unsupervised(self, csls, metrics, mode="csls"):
        max_src_word_considered = 10000
        _, metrics['unsupervised'] = self.get_match_samples(
            csls, np.arange(
                min(self.src_lang.vocab, int(max_src_word_considered))),
            1, mode=mode)
        self.logger.info(
            "{0:12s}: {1:5.4f}".format(
                "Unsupervised", metrics['unsupervised']))
        return metrics

    def monolingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'mono_dict'):
            mono_lingual_dir = os.path.join(self.data_dir, "monolingual")
            self.mono_dict = MonoDictionary(self.src_lang, mono_lingual_dir)
        if not self.mono_dict.atleast_one:
            return metrics
        metrics = self.mono_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info("{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
            "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['monolingual']:
            if dname == "mean":
                mean = metrics['monolingual'][dname]
                continue
            found = metrics['monolingual'][dname]['found']
            not_found = metrics['monolingual'][dname]['not_found']
            correlation = metrics['monolingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))
        return metrics

    def crosslingual(self, csls, metrics, **kwargs):
        if not hasattr(self, 'cross_dict'):
            cross_lingual_dir = os.path.join(self.data_dir, "crosslingual")
            self.cross_dict = CrossLingualDictionary(
                self.src_lang, self.tgt_lang, cross_lingual_dir)
        if not self.cross_dict.atleast_one:
            return metrics
        metrics = self.cross_dict.get_spearman_r(csls, metrics)
        self.logger.info("=" * 64)
        self.logger.info(
            "{0:>25s}\t{1:5s}\t{2:10s}\t{3:5s}".format(
                "Dataset", "Found", "Not Found", "Corr"))
        self.logger.info("=" * 64)
        mean = -1
        for dname in metrics['crosslingual']:
            if dname == "mean":
                mean = metrics['crosslingual'][dname]
                continue
            found = metrics['crosslingual'][dname]['found']
            not_found = metrics['crosslingual'][dname]['not_found']
            correlation = metrics['crosslingual'][dname]['correlation']
            self.logger.info(
                "{0:>25s}\t{1:5d}\t{2:10d}\t{3:5.4f}".format(
                    dname, found, not_found, correlation))
        self.logger.info("=" * 64)
        self.logger.info("Mean Correlation: {0:.4f}".format(mean))

    def evaluate(self, csls, evals, mode="csls"):
        """
        Evaluates the csls object on the functions specified in evals
            :param csls: CSLS object, which contains methods to
                         map source space to target space and such
            :param evals: list(str): The functions to evaluate on
            :return metrics: dict: The metrics computed by evaluate.
        """
        metrics = {}
        for eval_func in evals:
            assert hasattr(self, eval_func), \
                "Eval Function {0} not found".format(eval_func)
            metrics = getattr(self, eval_func)(csls, metrics, mode=mode)
        return metrics

    def get_match_samples(self, csls, range_indices, n,
                          use_mean=True, mode="csls", hubness_thresh=20):
        """
        Computes the n nearest neighbors for range_indices (from src)
        wrt the target in csls object.
        For use_mean True, this computes the avg metric for all
        top k nbrs across batch, and reports the average top 1 metric.
            :param csls : The csls object
            :param range_indices: The source words (from 0, ... )
            :param n: Number of nbrs to find
            :param use_mean (bool): Compute the mean or not
        """
        target_indices, metric = csls.get_closest_csls_matches(
            source_indices=range_indices,
            n=n, mode=mode)
        if use_mean:
            logger = logging.getLogger(__name__)
            # Hubness removal
            unique_tgts, tgt_freq = np.unique(target_indices,
                                              return_counts=True)
            hubs = unique_tgts[tgt_freq > hubness_thresh]
            old_sz = metric.shape[0]
            filtered_metric = metric[np.isin(
                target_indices, hubs, invert=True)]
            logger.info(
                "Removed {0:d} hub elements For unsupervised".format(
                    old_sz - filtered_metric.shape[0]))
            mean_metric = np.mean(filtered_metric)
            return target_indices, mean_metric
        else:
            return target_indices, metric

    def hubness(self, csls, source_range=None):
        """
        Computes the distribution of the number of words source words of which a particular target word is a 
        nearest neighbour based on simple cosine similarity / CSLS metric
        """
        if source_range is None:
            source_range = self.src_lang.vocab
        logger = logging.getLogger()
        logger.info('Computing Hubness')
        fig = plt.figure()
        ax = plt.gca()

        for use_nn in [True, False]:
            target_indices, _ = csls.get_closest_csls_matches(np.arange(source_range), 1, False, use_nn = use_nn)
            _, freq = np.unique(target_indices, return_counts=True)
            freq = [w for w in freq if w > 20 and w < 80]
            ax.hist(freq, bins = 200, label = 'CSLS' if use_nn == False else 'NN')
        plt.xlabel('hubness count')
        plt.ylabel('frequency')
        plt.title('Hubness Measure for different metrics')
        ax.legend(loc = 'best')
        plt.savefig(os.path.join(self.save_dir, 'hubness_graph'))
        import sys; sys.exit()


class Hubness(object):
    def __init__(self, src_lang, tgt_lang, csls, save_dir, use_nn):
        assert isinstance(src_lang, Language) and isinstance(tgt_lang, Language)
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.save_dir = save_dir
        self.use_nn = use_nn
        self.csls = csls
        self.target_matches, _ = self.csls.get_closest_csls_matches(source_indices=np.arange(self.src_lang.vocab),
                                                                    n=1,
                                                                    use_mean=False,
                                                                    use_nn=use_nn)
        self.target_matches = self.target_matches[:, 0]

    def get_hubs_np(self, thresh):
        """
            :param thresh: The frequency threshold for hubs
            :return hubs: The index of target words that are hubs
        """
        unique_tgts, tgt_freq = np.unique(self.target_matches, return_counts=True)
        hubs = unique_tgts[tgt_freq > thresh]
        return hubs

    def get_hubs(self, thresh):
        '''
        returns a list of target words which are hubs
        '''
        frequency = {}
        for i in self.target_matches:
            if i not in frequency:
                frequency[i] = 0
            frequency[i] += 1
        hubs = [w for w in frequency.keys() if frequency[w] > thresh]
        return [self.tgt_lang.ix2word[w] for w in hubs]

    def get_neighbours(self, word):
        '''
        returns the source words which have the given target word as their nearest neighbour
        '''
        word = self.tgt_lang.word2ix[word]
        neighbours = [i for i in range(self.src_lang.vocab) if self.target_matches[i] == word]
        return [self.src_lang.ix2word[w] for w in neighbours]

    def get_hub_dict(self, thresh):
        '''
        Dumps a dictionary consisting of hubs and their neighbours
        '''
        hubs = self.get_hubs(thresh)
        word_dict = set()
        for i in hubs:
            neighbours = self.get_neighbours(i)
            for j in neighbours:
                word_dict.add((i, j))
        with open(os.path.join(self.save_dir, 'hub_dict'), 'w') as f:
            for (i, j) in word_dict:
                f.write("{0:s}\t{1:s}\n".format(i, j))
        return word_dict


class GAN(object):
    """
    Takes in the discriminator, generator and other params and
    runs the training and evaluation. We transform from src -> tgt
    """
    def __init__(self, disc, gen, gpu, batcher, src, tgt, save_dir, data_dir=None, load_dir=None):
        """
        inputs:
            :param disc (Discriminator) : the Discriminator
            :param gen (Generator) : the Generator
            :param src (str): name of source lang
            :param tgt (str): name of target lang
            :param gpu (bool): Use the gpu
            :param batcher (``Batcher``): The batcher object
            :param data_dir (str): Location of data
            :param save_dir (str): Location to save models
        """
        self.src = src
        self.tgt = tgt
        self.gen = gen.cuda() if gpu else gen
        self.disc = disc.cuda() if gpu else disc

        self.batcher = batcher
        self.gpu = gpu
        data_dir = 'data' if data_dir is None else data_dir
        self.evaluator = Evaluator(self.batcher[self.src], self.batcher[self.tgt], data_dir, save_dir)
        self.best_metrics = None
        self.save_dir = save_dir
        self.best_eval_metric = -1.0
        self.best_model_state = None
        self.best_supervised_metric = -1.0
        if load_dir is not None:
            checkpoint_path = os.path.join(load_dir, 'best_model', 'checkpoint.pth.tar')
            checkpoint = torch.load(checkpoint_path)
            self.gen.load_state_dict(checkpoint['gen_params'])
            self.disc.load_state_dict(checkpoint['disc_params'])
            self.gen = self.gen.cuda() if gpu else gen
            self.disc = self.disc.cuda() if gpu else disc
            self.log(0, None)

    def supervised_rcsls_loss(self, batch_sz, k=10, num_tgts=50000):
        # first an assert to ensure unit norming
        if not hasattr(self, "check_rcsls_valid"):
            self.check_rcsls_valid = True
            for l in self.batcher.name2lang.values():
                if l.unit_norm is False:
                    self.check_rcsls_valid = False
                    break
        if not self.check_rcsls_valid:
            raise RuntimeError("For RCSLS, need to unit norm")
        src, tgt, nn_src, nn_tgt = self.batcher.supervised_rcsls_minibatch(
            batch_sz, self.src, self.tgt, num_tgts)
        xtrans = self.gen(Variable(src))
        yvar = Variable(tgt)
        sup_loss = 2 * torch.sum(xtrans * yvar)
        # Compute nearest nn loss wrt src
        nn_tgt = Variable(nn_tgt)
        dmat = torch.mm(xtrans, nn_tgt.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_tgt[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)  # Detach from compute graph
        nnloss = torch.bmm(nnbrs, xtrans.unsqueeze(-1)).squeeze(-1)
        nn_tgt_loss = torch.sum(nnloss) / k
        # Compute nearest nn loss wrt tgt
        nn_src = Variable(nn_src)
        nn_src_transform = Variable(self.gen(nn_src).data)
        dmat = torch.mm(yvar, nn_src_transform.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_src[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)
        nnloss = torch.bmm(self.gen(nnbrs), yvar.unsqueeze(-1)).squeeze(-1)
        nn_src_loss = torch.sum(nnloss) / k
        return - (sup_loss - nn_tgt_loss - nn_src_loss) / src.size(0)

    def supervised_loss(self, batch_sz, sup_factor=1.):
        src_emb, tgt_emb = self.batcher.supervised_minibatch(batch_sz, self.src, self.tgt)
        preds = self.gen(Variable(src_emb))
        loss = -sup_factor * torch.mean(F.cosine_similarity(preds, Variable(tgt_emb)))
        return loss

    def discriminator_loss(self, batch_sz, smoothing):
        self.disc.train()
        batch = self.batcher.minibatch(batch_sz)  # {lang: (idx (batch_size), batch_size x n_dim)}
        src, tgt = batch[self.src][1], batch[self.tgt][1]
        src, tgt = Variable(src), Variable(tgt)  # batch x n_dim, batch x n_dim
        src = Variable(self.gen(src).data)  # batch x n_dim
        inp = torch.cat([src, tgt], 0)
        preds = self.disc(inp).squeeze(-1)
        tgts = torch.zeros_like(preds)
        tgts[:src.size(0)] = smoothing
        tgts[src.size(0):] = 1. - smoothing
        return F.binary_cross_entropy(preds, tgts)

    def auto_loss(self, batch_sz, **kwargs):
        _, batch = self.batcher[self.src].minibatch(batch_sz)
        encode = self.gen(Variable(batch))
        decode = F.linear(encode, self.gen.transform.weight.t(), None)
        loss = -F.cosine_similarity(decode, Variable(batch)).mean()
        return loss

    def noisy_auto_loss(self, batch_sz, std):
        if not hasattr(self, 'noise'):
            self.noise = GaussianAdditive(gpu=self.gpu, std=std)
        _, batch = self.batcher[self.src].minibatch(batch_sz)
        encode = self.gen(self.noise.generate(Variable(batch)))
        decode = F.linear(encode, self.gen.transform.weight.t(), None)
        loss = -F.cosine_similarity(decode, Variable(batch)).mean()
        return loss

    def generator_loss(self, batch_sz, smoothing):
        self.disc.eval()
        src = self.batcher[self.src].minibatch(batch_sz)[1]  # batch x n_dim
        src = Variable(src)
        fake_tgt = self.gen(src)  # batch x n_dim
        disc_fake = self.disc(fake_tgt).squeeze()  # batch,
        real_targets = torch.ones_like(disc_fake) - smoothing
        return F.binary_cross_entropy(disc_fake, real_targets)

    def train_artetxe(self, pairs, procrustes_iters,
            procrustes_dict_size=0,
            procrustes_tgt_rank=15000,
            procrustes_thresh=0.,
            hubness_thresh=20,
            epochs=1, mode="csls", save=True, eval_metric='unsupervised'):
        # First procrustes with dictionary
        weight = self.procrustes_onestep(pairs)
        self.gen.transform.weight.data.copy_(weight)
        self.log(0, save=save, mode=mode, eval_metric=eval_metric)
        self.best_model_state = None
        # Now the model has been set
        self.procrustes(procrustes_iters,
            procrustes_dict_size=0,
            procrustes_tgt_rank=15000,
            procrustes_thresh=0.,
            epochs=1, mode=mode, save=save,
            hubness_thresh=hubness_thresh)
        return self.best_supervised_metric

    def train_rcsls(
        self, pairs, mode, save=True,
        niter=10, spectral=False, k=10, num_tgts=50000,
        opt_params={"name": "SGD", "lr": 1.0, "momentum": 0.},
        batch_size=-1, logafter=500, eval_metric='unsupervised'
    ):
        # train rcsls
        # Initialize with procrustes
        logger = logging.getLogger(__name__)
        weight = self.procrustes_onestep(pairs)
        self.gen.transform.weight.data.copy_(weight)
        self.log(0, save=False, mode=mode, eval_metric=eval_metric)
        self.best_model_state = None
        # Now model has been set. Let's start with RCSLS
        name = opt_params.pop("name")
        optimizer = getattr(optim, name)(self.gen.parameters(), **opt_params)
        fold = np.inf
        for it in range(niter + 1):
            if opt_params["lr"] < 1e-4:
                break
            optimizer.zero_grad()
            loss = self.supervised_rcsls_loss(
                batch_size, k=k, num_tgts=num_tgts)
            f = loss.item()
            lr_str = opt_params["lr"]
            logger.info(
                f"Iteration: {it + 1}, Learning Rate: {lr_str}, Loss: {f}")
            if f > fold and batch_size == -1:
                opt_params["lr"] /= 2
                optimizer = getattr(optim, name)(
                    self.gen.parameters(), **opt_params)
                f = fold
            else:
                loss.backward()
                optimizer.step()
                if spectral is True:
                    self.gen.spectral()
            if logafter > 0:
                if it % logafter == 0:
                    self.log(it, save=False, mode=mode, eval_metric=eval_metric)
            else:
                self.log(it, save=False, mode=mode, eval_metric=eval_metric)
            fold = f

    def train(self, epochs, iters_per_epoch, batch_sz, opt,
        opt_params, sup_opt, factor={"sup": 1., "unsup": 1.,"ortho": 1.},
        smoothing=0.0, log_after=5000, lr_decay=0.98, ortho_params=None,
        num_disc_rounds=1, num_gen_rounds=1, eval_batches=500, orthogonal="",
        supervised_method="cosine", num_supervised_rounds=0, num_nbrs=50000, k=10,
        procrustes_iters=0, procrustes_dict_size=0, procrustes_tgt_rank=15000, 
        procrustes_thresh=0., eval_metric='unsupervised', lr_local_dk=0.5,
        patience=2):
        """
            :param epochs (int) : # epochs
            :param iters_per_epoch (int): Iterations per epoch
            :param batch_sz (int) : The batch size
            :param opt (str) : The optimizer name (Adam, RMSprop etc.)
            :param opt_params (dict) : The parameters
            :param sup_opt (dict) : The optimizer parameters for supervised loss
            :param factor (dict) : The factors weighing the supervised, unsupervised and the
                auto loss.
            :param smoothing (float) : Label smoothing
            :param log_after (int) : Log losses after how many steps
            :param lr_decay (float) : Learning rate decaying for the learning
            :param ortho_params (dict) : The orthogonalization params 
            :param num_disc_rounds (int) : Rounds of training for discriminator
            :param num_gen_rounds (int) : Rounds of training for generator
            :param eval_batches (int): Evaluate batches after how many iterations
            :param orthogonal (str) : Orthogonalize the map matrix W
            :param supervised_method (str) : The supervised loss to use
                (cosine match or RCSLS loss)
            :param num_supervised_rounds (int) : Number of rounds of supervised updates
            :param num_nbrs (int) : Number of NN used for RCSLS computation in supervised loss
            :param k (int) : Number of neighbors for CSLS computation
            :param procrustes_iters (int) : Number of rounds of procrustes to do
            :param procrustes_dict_size (int) : Maximum number of elements to use during
                the dictionary expansion
            :param procrustes_tgt_rank (int) : Number of examples from the target dictionary
                to use for procrustes
            :param procrustes_thresh (float) : Threshold to use for csls metric while computing
                procrustes
            :param eval_metric (str) : The metric used for lr decay
            :param lr_local_dk (float) : Decaying of LR by this factor when we observe bad runs > patience
                for the parameter as specified by eval_metric
            :param patience (int) : Epochs to wait for without improvement in eval_metric to decay LR
        """
        self.disc_opt = getattr(optim, opt)(self.disc.parameters(), **opt_params)
        self.gen_opt = getattr(optim, opt)(self.gen.parameters(), **opt_params)
        if sup_opt == 'Adam':
            self.gen_sup_opt = getattr(optim, sup_opt)(self.gen.parameters())
        else:
            self.gen_sup_opt = getattr(optim, sup_opt)(self.gen.parameters(), **opt_params)
            
        factor_sum = sum(factor.values())
        factor = {x: factor[x] / float(factor_sum) for x in factor}
        # For 1 discriminator and generator update, 1.5 batches of src and 1 batches of tgt are used.

        scheduler_gen = MonitorLR(name='GEN', optimizer=self.gen_opt, min_lr=1e-5, factor=lr_decay, shrink=lr_local_dk, patience=patience)
        batch_gen_loss = []
        batch_disc_loss = []
        batch_sup_loss = []
        batch_auto_loss = []
        logger = logging.getLogger(__name__)
        for epoch in range(epochs):
            # Run the training loop, gather data
            start = time.time()
            for counter, itr in enumerate(range(0, iters_per_epoch, batch_sz)):
                for _ in range(num_gen_rounds):
                    self.gen.zero_grad()
                    gen_loss = self.generator_loss(batch_sz, smoothing)
                    gen_loss *= factor["unsup"]
                    gen_loss.backward()
                    self.gen_opt.step()
                    batch_gen_loss.append(gen_loss.item())
                for _ in range(num_supervised_rounds):
                    self.gen_sup_opt.zero_grad()
                    if supervised_method == "cosine":
                        sup_loss = self.supervised_loss(batch_sz)
                    elif supervised_method == "rcsls":
                        sup_loss = self.supervised_rcsls_loss(
                            batch_sz, k=k, num_tgts=num_nbrs)
                    else:
                        raise NotImplementedError(
                            f"{supervised_method} not implemented")
                    sup_loss *= factor["sup"]
                    sup_loss.backward()
                    self.gen_sup_opt.step()
                    batch_sup_loss.append(sup_loss.item())
                if orthogonal != '':
                    if orthogonal == "orthogonalize":
                        self.gen.orthogonalize(**ortho_params)
                    else:
                        self.gen.zero_grad()
                        auto_loss = getattr(self, orthogonal)(batch_sz, **ortho_params)
                        auto_loss *= factor["ortho"]
                        auto_loss.backward()
                        self.gen_opt.step()
                        batch_auto_loss.append(auto_loss.item())
                for _ in range(num_disc_rounds):
                    self.disc.zero_grad()
                    disc_loss = self.discriminator_loss(batch_sz, smoothing)
                    disc_loss.backward()
                    self.disc_opt.step()
                    batch_disc_loss.append(to_numpy(disc_loss, self.gpu).item())

                if itr % log_after == 0:
                    writebuf = "{0:12s}: {1:5d}, Epoch: {2:4d}, Gen Loss: {3:7.4f}, Desc Loss: {4:7.4f}".format("Iteration", itr, epoch, np.mean(batch_gen_loss), np.mean(batch_disc_loss))
                    del batch_gen_loss[:]
                    del batch_disc_loss[:]
                    if len(batch_sup_loss) > 0:
                        writebuf += ", Sup Loss: {0:7.4f}".format(np.mean(batch_sup_loss))
                        del batch_sup_loss[:]
                    if len(batch_auto_loss) > 0:
                        writebuf += ", Auto Loss: {0:7.4f}".format(np.mean(batch_auto_loss))
                        del batch_auto_loss[:]
                    logger.info(writebuf)
            evaluation_metric = self.log(epoch + 1, ["supervised", "unsupervised", "monolingual"], eval_metric=eval_metric)
            scheduler_gen.step(evaluation_metric)
            logger.info("Finished epoch ({0:d} / {1:d}). Took {2:.2f}s".format(epoch + 1, epochs, time.time() - start))
        logger.info("Finished Training after {0} epochs".format(epochs))
        logger.info("{0:12s}: {1:5.4f}".format("Unsupervised", self.best_metrics['unsupervised']))
        logger.info("Found {0:d} words for supervised metric. Precision@1: {1:5.2f}\t@5: {2:5.2f}\t@10: {3:5.2f}".format(int(self.best_metrics['total']),
            self.best_metrics['acc1'], self.best_metrics['acc5'], self.best_metrics['acc10']))
        self.procrustes(procrustes_iters, procrustes_dict_size, procrustes_tgt_rank, procrustes_thresh, epochs=epochs, eval_metric=eval_metric)

    def expand_dict(self, procrustes_dict_size=0., procrustes_tgt_rank=15000,
        procrustes_thresh=0., mode="csls", hubness_thresh=20):
        csls = self.get_csls()
        logger = logging.getLogger()
        num_source_words = self.batcher.name2lang[self.src].vocab
        metric_source_range = np.arange(num_source_words)
        assert self.batcher.name2lang[self.tgt].vocab >= 10
        indices, metrics = self.evaluator.get_match_samples(csls, metric_source_range, 2, use_mean=False, mode=mode)
        # Filtering
        # 1. clear out words in the target.
        indices = indices[:, 0]
        pairs = np.concatenate([np.arange(indices.shape[0])[:, np.newaxis], indices[:, np.newaxis]], axis=1)
        if procrustes_tgt_rank > 0:
            filter_ix = indices < procrustes_tgt_rank
            pairs = pairs[filter_ix]
            metrics = metrics[filter_ix]
        diff = metrics[:, 0] - metrics[:, 1]
        if procrustes_thresh > 0:
            filter_ix = diff > procrustes_thresh
            pairs = pairs[filter_ix]
            metrics = metrics[filter_ix]
        if procrustes_dict_size > 0:
            sorted_indices = np.argsort(diff)[-procrustes_dict_size:]
            pairs = pairs[sorted_indices]
            metrics = metrics[sorted_indices]
        if hubness_thresh > 0:
            unique_tgts, tgt_freq = np.unique(indices, return_counts=True)
            hubs = unique_tgts[tgt_freq > hubness_thresh]
            old_pairs_sz = pairs.shape[0]
            pairs = pairs[np.isin(pairs[:, 1], hubs, invert=True)]
            logger.info("Removed {0} hub elements".format(old_pairs_sz - pairs.shape[0]))
        return pairs

    def procrustes(self, procrustes_iters, procrustes_dict_size=0.,
        procrustes_tgt_rank=15000, procrustes_thresh=0., epochs=10, mode="csls",
        save=True, hubness_thresh=20, eval_metric="unsupervised"):
        logger = logging.getLogger()
        for itr in range(procrustes_iters):
            if self.best_model_state is not None:
                self.gen.load_state_dict(self.best_model_state)
            logger.info("Refining using Procrustes. ({0} / {1})".format(itr + 1, procrustes_iters))
            pairs = self.expand_dict(procrustes_dict_size=procrustes_dict_size,
                                     procrustes_tgt_rank=procrustes_tgt_rank,
                                     procrustes_thresh=procrustes_thresh,
                                     mode=mode,
                                     hubness_thresh=hubness_thresh)
            weight = self.procrustes_onestep(pairs)
            self.gen.transform.weight.data.copy_(weight)
            self.log(epochs, save=save, mode=mode, model_file="procrustes-checkpoint.pth.tar", metrics_file="procrustes-metrics.json", eval_metric=eval_metric)

    def procrustes_onestep(self, pairs):
        src_aligned_embeddings = self.batcher[self.src].embeddings[to_cuda(torch.LongTensor(pairs[:, 0]), self.gpu)]  # num_procrustes x dim_sz
        tgt_aligned_embeddings = self.batcher[self.tgt].embeddings[to_cuda(torch.LongTensor(pairs[:, 1]), self.gpu)]  # num_procrustes x dim_sz
        matrix = torch.mm(tgt_aligned_embeddings.transpose(1, 0), src_aligned_embeddings)
        u, _, v = torch.svd(matrix)
        weight = torch.mm(u, v.t())
        return weight

    def get_checkpoint(self, metrics):
        """
            Sends back the checkpoint which stores relevant information about the model
            :param metrics (dict: str -> float): The dictionary of metrics
        """
        assert all(metric in metrics for metric in ["acc1", "acc5", "acc10", "unsupervised", "total"]), "Not all metrics found"
        checkpoint = OrderedDict()
        for metric in metrics:
            checkpoint[metric] = metrics[metric]
        checkpoint['gen_params'] = self.gen.state_dict()
        checkpoint['disc_params'] = self.disc.state_dict()
        if self.gpu:
            checkpoint['gen_params'] = {param: checkpoint['gen_params'][param].cpu() for param in checkpoint['gen_params']}
            checkpoint['disc_params'] = {param: checkpoint['disc_params'][param].cpu() for param in checkpoint['disc_params']}
        if hasattr(self, 'gen_opt'):
            checkpoint['gen_optimizer_stats'] = self.gen_opt.state_dict()
            checkpoint['disc_optimizer_stats'] = self.disc_opt.state_dict()
        return checkpoint

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        if hasattr(self, 'gen_opt'):
            self.gen_opt.load_state_dict(checkpoint['gen_optimizer_stats'])
            self.disc_opt.load_state_dict(checkpoint['disc_optimizer_stats'])
        self.gen.load_state_dict(checkpoint['gen_params'])
        self.disc.load_state_dict(checkpoint['disc_params'])
        if self.gpu:
            self.gen.cuda()
            self.disc.cuda()

    def log(self, n_iter, eval_list=["supervised", "unsupervised", "monolingual"], save=True, eval_metric='unsupervised', **kwargs):
        csls = self.get_csls()
        if eval_list is None:
            hub_eval = Hubness(self.batcher.name2lang[self.src], self.batcher.name2lang[self.tgt], csls, self.save_dir, False)
            return
        metrics = self.evaluator.evaluate(csls, eval_list, mode=kwargs['mode'] if 'mode' in kwargs else 'csls')
        if metrics[eval_metric] > self.best_eval_metric:
            logger = logging.getLogger(__name__)
            new_metric = metrics[eval_metric]
            logger.info(f"Metric {eval_metric} improved from {self.best_eval_metric:2.2f} to {new_metric:2.2f}")
            self.best_eval_metric = metrics[eval_metric]
            self.best_metrics = {metric: np.float64(metrics[metric]) if metric != 'monolingual' else metrics[metric] for metric in metrics}
            self.best_supervised_metric = metrics['acc1']
            self.best_model_state = deepcopy(self.gen.state_dict())
            if save:
                # Save model
                checkpoint = self.get_checkpoint(metrics)
                model_file = 'checkpoint.pth.tar' if 'model_file' not in kwargs else kwargs['model_file']
                checkpoint_path = os.path.join(self.save_dir, 'best_model', model_file)
                torch.save(checkpoint, checkpoint_path)
                # Save metrics
                metrics_file = 'metrics.json' if 'metrics_file' not in kwargs else kwargs['metrics_file']
                metrics_path = os.path.join(self.save_dir, 'best_model', metrics_file)
                with open(metrics_path, 'w') as fp:
                    json.dump(self.best_metrics, fp)
        return metrics[eval_metric]

    def get_csls(self):
        source_word_embeddings = self.batcher.name2lang[self.src].embeddings
        dest_word_embeddings = self.batcher.name2lang[self.tgt].embeddings
        return CSLS(src=source_word_embeddings,
                    tgt=dest_word_embeddings,
                    map_src=self.gen,
                    gpu=self.gpu)
