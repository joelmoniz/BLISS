from __future__ import absolute_import, print_function, division
import io
import numpy as np
import warnings
import os

import torch
import torch.autograd
from scipy.stats import spearmanr
import logging

from bliss.utils import make_directory, to_cuda

'''
Class Language, creates and loads data for 1 language
Creates the vocabulary and inverse vocabulary
store language metadata
'''


class Language(object):
    def __init__(self, name, gpu=False, mode='seq', mean_center=False, unit_norm=False):
        """
        inputs
            :param name (str): The name of the vocab language
            :param mode (str): Random or epoch-wise sampling ['seq', 'rand']
        Attributes
            :embeddings (numpy array later) : |V| x embed_dim
            :ix2word (list) : Word index, based on frequency
            :word2ix (dict) : Word -> idx
            :vocab (int) : Size |V|
            :perm (internal) : The shuffled indices
            :ct (int) : current counter (pointer to start the next batch)
            :epoch (int) : epoch counter
        """
        self.name = name
        self.embeddings = []
        self.ix2word = []
        self.word2ix = {}
        self.vocab = 0
        self._perm = None
        self.ct = 0
        self.epoch = 0
        self.max_freq = -1
        self.mode = mode
        self.gpu = gpu
        self.mean_center = mean_center
        self.unit_norm = unit_norm

    def __getitem__(self, i):
        self.ix2word[i]

    def __contains__(self, w):
        return w in self.word2ix

    def index(self, word):
        return self.word2ix[word]

    def load(self, file, dir_name, max_freq=-1, max_count=200000, init_norm=True):
        """
        Loads the file (word 300 dim embedding) (the first line is the name. Ignore)
            :param file (str) : file name
            :param dir_name (str) : the directory from where data is located
            :returns None
        """
        folder = os.path.join(dir_name, file) + '_dir'
        file = os.path.join(dir_name, file)
        if os.path.exists(folder):
            embeddings_file = os.path.join(folder, 'embeddings.npy')
            ix2word_file = os.path.join(folder, 'ix2word.npy')
            assert os.path.exists(embeddings_file), "Embedding file not found at %s" % (embeddings_file)
            assert os.path.exists(ix2word_file), "Vocab index file not found at %s" % (ix2word_file)
            self.embeddings = np.load(embeddings_file)
            self.ix2word = np.load(ix2word_file)
        else:
            embeddings = []
            word_count = 0
            with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                start_line = True
                for ix, linex in enumerate(f.readlines()):
                    if start_line:
                        start_line = not start_line
                        continue
                    word, vec = linex.rstrip().split(' ', 1)
                    vect = np.fromstring(vec, sep=' ')
                    if len(word) == 0 or vect.shape[0] < 300:
                        print('Skipping at', ix)
                        continue
                    self.ix2word.append(word)
                    embeddings.append(vect)
                    word_count += 1
                    if word_count == max_count:
                        break
            self.ix2word = np.array(self.ix2word)
            self.embeddings = np.array(embeddings)
            make_directory(folder)
            np.save(os.path.join(folder, 'embeddings.npy'), self.embeddings)
            np.save(os.path.join(folder, 'ix2word.npy'), self.ix2word)

        self.embeddings = to_cuda(torch.from_numpy(self.embeddings).float(), self.gpu)
        logger = logging.getLogger(__name__)
        if init_norm:
            logger.info("Unit Norming")
            self.embeddings.div_(self.embeddings.norm(2, 1, keepdim=True))
        if self.mean_center:
            logger.info("Mean Centering")
            self.embeddings.sub_(self.embeddings.mean(0, keepdim=True))
        # if self.unit_norm:
            logger.info("Unit Norming")
            self.embeddings.div_(self.embeddings.norm(2, 1, keepdim=True))
        self.vocab = len(self.ix2word)
        self.max_freq = self.vocab - 1 if max_freq == -1 else min(max_freq, self.vocab - 1)
        self.word2ix = {self.ix2word[i]: i for i in range(self.vocab)}
        if self.mode == 'seq':
            self._perm = np.random.permutation(self.max_freq + 1)

    def minibatch(self, batch_sz):
        """
        Returns a minibatch of fixed size
            :param batch_sz (int) : The batch size
            :returns batch : (np.array(batch_sz,), np.array(batch_sz x embed_dim))
        """
        if self.mode == 'seq':
            idx = self._perm[self.ct: self.ct + batch_sz]
            if len(idx) < batch_sz:
                idx = np.concatenate((idx, self._perm[: batch_sz - len(idx)]))
            self.ct += batch_sz
            if self.ct >= self.vocab:
                self._perm = np.random.permutation(self.vocab)
                self.epoch += 1
                self.ct %= self.vocab
        else:
            # idx = np.random.randint(0, self.max_freq + 1, size=(batch_sz))
            # idx = torch.LongTensor(idx)
            idx = torch.LongTensor(batch_sz).random_(self.max_freq + 1)
        return (idx, self.embeddings[to_cuda(idx, self.gpu)])

    def __len__(self):
        """
            Length of vocab
        """
        return self.vocab

    def __str__(self):
        """
            The name of the vocab
        """
        return self.name

    def get_embeddings(self, idx):
        idx = to_cuda(torch.LongTensor(idx), self.gpu)
        return self.embeddings[idx]


class Batcher(object):
    def __init__(self, languages):
        """
            inputs
                languages (tuple of Language objects)
        """
        self.languages = languages
        self.num_languages = len(self.languages)
        self.name2lang = {i.name: i for i in self.languages}

    def __getitem__(self, lang):
        return self.name2lang[lang]

    def minibatch(self, batch_sz):
        """
        If there are k languages then return a dictionary with keys
        as langauge names and values as batch_sz / k samples of
        each langauge.
            :param batch_sz (int): Batch size for mixture
            :returns dict (key: language name, value: (ix, embedding)) : (batch_size // k, batch_size // k * n_dim)
        """
        ret = {k.name: k.minibatch(batch_sz) for k in self.languages}
        return ret

    def load_from_supervised(self, fil, src, tgt, dir_name, max_count=5000):
        """
            Loads from supervised file.
            :fil : filename
            :dir_name: The directory name
            :max_count: how many words to use
        """
        if not hasattr(self, 'pair2ix'):
            self.pair2ix = {}
        self.pair2ix["{0:s}-{1:s}".format(src, tgt)] = WordDictionary(self.name2lang[src], self.name2lang[tgt], os.path.join(dir_name, fil))
        if max_count > 0:
            self.pair2ix["{0:s}-{1:s}".format(src, tgt)].word_map = self.pair2ix["{0:s}-{1:s}".format(src, tgt)].word_map[:max_count]

    def expand_supervised(self, gan, src, tgt, train_params):
        assert hasattr(self, 'pair2ix'), "Need supervised dict to expand"
        key = "{0:s}-{1:s}".format(src, tgt)
        weight = gan.procrustes_onestep(self.pair2ix[key].word_map)
        gan.gen.transform.weight.data.copy_(weight)
        pairs = gan.expand_dict(
            procrustes_dict_size=train_params['procrustes_dict_size'],
            procrustes_tgt_rank=train_params['procrustes_tgt_rank'],
            procrustes_thresh=train_params['procrustes_thresh']
        )
        old_size = self.pair2ix[key].word_map.shape[0]
        self.pair2ix[key].word_map = pairs
        gan.gen._initialize()
        new_size = self.pair2ix[key].word_map.shape[0]
        logger = logging.getLogger()
        logger.info("Expanded dictionary from {0} -> {1}".format(
            old_size, new_size))

    def supervised_minibatch(self, batch_sz, src, tgt):
        key = "{0:s}-{1:s}".format(src, tgt)
        if batch_sz > 0:
            idx = np.random.randint(
                self.pair2ix[key].word_map.shape[0], size=(batch_sz,))
            src_tgt_pairs = self.pair2ix[key].word_map[idx]
        else:
            src_tgt_pairs = self.pair2ix[key].word_map
        src_emb = self.name2lang[src].get_embeddings(src_tgt_pairs[:, 0])
        tgt_emb = self.name2lang[tgt].get_embeddings(src_tgt_pairs[:, 1])
        return src_emb, tgt_emb

    def supervised_rcsls_minibatch(self, batch_sz, src, tgt, num_ex):
        src_emb, tgt_emb = self.supervised_minibatch(batch_sz, src, tgt)
        nn_src = self.name2lang[src].get_embeddings(np.arange(num_ex))
        nn_tgt = self.name2lang[tgt].get_embeddings(np.arange(num_ex))
        return src_emb, tgt_emb, nn_src, nn_tgt


class WordDictionary(object):
    def __init__(self, src, tgt, filename):
        '''
        src: Language object for the source language
        tgt: Language object for the target object
        data_dir: The directory where data is located
        Assumes that the filename is data_dir/<src-target>
        '''
        self.logger = logging.getLogger()
        if not os.path.exists(filename + '.npy'):
            word_map = []
            not_found = 0
            total = 0
            src_not_found = 0
            tgt_not_found = 0
            with io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                for line in f:
                    lin = line.strip().split()
                    if len(lin) != 2:
                        continue
                    total += 1
                    src_word, tgt_word = lin[0], lin[1]
                    if src_word not in src.word2ix:
                        src_not_found += 1
                        not_found += 1
                    elif tgt_word not in tgt.word2ix:
                        tgt_not_found += 1
                        not_found += 1
                    else:
                        word_map.append([src.word2ix[src_word], tgt.word2ix[tgt_word]])
            word_map = np.array(word_map, dtype=int)
            self.logger.info("({0:d} / {1:d}) not found ({2:d} src not found and {3:d} tgt not found).".format(not_found, total, src_not_found, tgt_not_found))
            np.save(arr=word_map, file=filename + ".npy")
        else:
            word_map = np.load(filename + ".npy")
        self.word_map = word_map
        self.src = src
        self.tgt = tgt


    def precisionatk(self, pred_words, klist):
        """
            :param pred_words: numpy array containing the idx of the tgt words (n, k)
            :param klist: list(int): precision at each element of klist is computed
            :return retvals: list(float): The precision at k (size(retvals) == size(klist))
            :return total: (int): The total unique elements
        # I prefer list comprehension over map functions
        I prefer vectorized operations to list comprehensions :P
        """
        max_k = max(klist)
        assert max_k <= pred_words.shape[1]
        assert min(klist) > 0
        retvals = []
        unique_w, indices = np.unique(self.word_map[:, 0], return_inverse=True)
        targets = np.tile(self.word_map[:, 1:], (1, max_k))
        for k in klist:
            acc = (targets[:, :k] == pred_words[:, :k]).sum(1)
            # Handle multiple translations
            _correct = np.zeros_like(unique_w)
            np.add.at(_correct, indices, acc)
            incorrect = (_correct == 0).sum() / _correct.size
            retvals.append((1. - incorrect) * 100.)
        return retvals, unique_w.size

    def precisionatk_nltk(self, pred_words, klist):
        '''
        precision at k function which takes into account polysemy
        using NLTK wordmap
        '''
        try:
            import nltk
            nltk.data.path = ['./nltk_data']
            from nltk.corpus import wordnet as wn
        except ImportError:
            raise RuntimeError("Need NLTK for this function.")

        nltk_map = {'es': 'spa', 'fr': 'fra', 'it': 'ita', 'en': 'eng', 'zh': 'cmn'}

        def set_correct(correct, val, prediction):
            for i, k in enumerate(klist):
                if len(set(correct) & set(prediction[:k])) > 0:
                    val[i] = 1

        ret_val = 1. * np.zeros_like(klist)

        word_map = {}
        for idx, (src, gold) in enumerate(self.word_map):
            if src not in word_map:
                word_map[src] = ([], pred_words[idx])
            word_map[src][0].append(gold)
        d = len(word_map)

        for word in word_map:
            prediction = self.tgt.ix2word[word_map[word][1]]
            val = np.zeros_like(klist)
            src_word = self.src.ix2word[word]
            gold = self.tgt.ix2word[word_map[word][0]]
            '''
            Normal Dictionary Matching
            '''
            set_correct(gold, val, prediction)
            '''
            Checking if any sense of the gold word matches with the prediction
            '''

            if self.tgt.name not in nltk_map:
                ret_val += val
                continue
            tgt_lang = nltk_map[self.tgt.name]
            synsets = [w for gold_word in gold for w in wn.synsets(gold_word)]
            similar_words = [w for synset in synsets for w in synset.lemma_names(tgt_lang)]
            set_correct(similar_words, val, prediction)

            '''
            Checking if the prediction is the translation of any sense of the source word
            '''

            if self.src.name not in nltk_map:
                ret_val += val
                continue
            synsets = wn.synsets(src_word)
            similar_words = [w for synset in synsets for w in synset.lemma_names(tgt_lang)]
            set_correct(similar_words, val, prediction)

            ret_val += val

        ret_val *= (100. / d)
        return ret_val, len(set(self.word_map[:, 0]))


class MonoDictionary(object):
    def __init__(self, lang, data_dir):
        monolingual_dir = os.path.join(data_dir, lang.name)
        self.datasets = {}
        self.atleast_one = False
        self.lang = lang
        if os.path.isdir(monolingual_dir):
            for fil in os.listdir(monolingual_dir):
                if fil.startswith(lang.name):
                    self.datasets[fil.rstrip('.txt')] = self.process(os.path.join(monolingual_dir, fil))
                    self.atleast_one = True
        else:
            warnings.warn("No monolingual dictionary found at {}. Skipping.".format(monolingual_dir))

    def process(self, path):
        indices = []
        values = []
        dataset = {}
        dataset['not_found'] = 0
        dataset['found'] = 0
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
            for line in fp:
                line = line.lower()
                line = line.strip().split()
                if len(line) != 3:
                    assert len(line) > 3, "Something wrong with preprocessing. Found {0} of length {1} in {2}".format(line, len(line), path)
                    assert 'SEMEVAL' in path or 'EN-IT_MWS353' in path, "Error. File {0} is not supposed to contain phrases".format(path)
                    continue
                if line[0] in self.lang.word2ix and line[1] in self.lang.word2ix:
                    dataset['found'] += 1
                    i1 = self.lang.word2ix[line[0]]
                    i2 = self.lang.word2ix[line[1]]
                    indices.append([i1, i2])
                    values.append(float(line[2]))
                else:
                    dataset['not_found'] += 1
        indices = np.array(indices, dtype=int)
        dataset['gold_scores'] = np.array(values)
        dataset['word1'] = indices[:, 0]
        dataset['word2'] = indices[:, 1]
        return dataset

    def get_spearman_r(self, csls, metrics):
        metrics['monolingual'] = {}
        for dname in self.datasets:
            dataset = self.datasets[dname]
            e1, e2 = csls.map_to_tgt(dataset['word1']), csls.map_to_tgt(dataset['word2'])
            score = (e1 * e2).sum(-1)
            correlation = spearmanr(score, dataset['gold_scores']).correlation
            metrics['monolingual'][dname] = {'correlation': correlation,
                                             'found': dataset['found'],
                                             'not_found': dataset['not_found']}
        metrics['monolingual']['mean'] = np.array([metrics['monolingual'][dname]['correlation'] for dname in metrics['monolingual']]).mean()
        return metrics


class CrossLingualDictionary(object):
    def __init__(self, src_lang, tgt_lang, data_dir):
        crosslingual_dir = os.path.join(data_dir, 'wordsim')
        self.datasets = {}
        self.atleast_one = False
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.flip = False
        if os.path.isdir(crosslingual_dir):
            for fil in os.listdir(crosslingual_dir):
                languages = fil.split('-')
                if languages[:2] == [self.src_lang.name, self.tgt_lang.name]:
                    pass
                elif languages[:2] == [self.tgt_lang.name, self.src_lang.name]: 
                    self.flip = True
                else:
                    continue
                self.datasets[fil.rstrip('.txt')] = self.process(os.path.join(crosslingual_dir, fil))
                self.atleast_one = True
        else:
            warnings.warn("No crosslingual dictionary found at {}. Skipping.".format(crosslingual_dir))

    def process(self, path):
        indices = []
        values = []
        dataset = {}
        dataset['not_found'] = 0
        dataset['found'] = 0
        if self.flip:
            word1, word2 = [1, 0]
        else:
            word1, word2 = [0, 1]
        with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
            for line in fp:
                line = line.lower()
                line = line.strip().split()
                if len(line) != 3:
                    assert len(line) > 3, "Something wrong with preprocessing. Found {0} of length {1} in {2}".format(line, len(line), path)
                    assert 'SEMEVAL' in path or 'EN-IT_MWS353' in path, "Error. File {0} is not supposed to contain phrases".format(path)
                    continue
                if line[word1] in self.src_lang.word2ix and line[word2] in self.tgt_lang.word2ix:
                    dataset['found'] += 1
                    i1 = self.src_lang.word2ix[line[word1]]
                    i2 = self.tgt_lang.word2ix[line[word2]]
                    indices.append([i1, i2])
                    values.append(float(line[2]))
                else:
                    dataset['not_found'] += 1
        indices = np.array(indices, dtype=int)
        dataset['gold_scores'] = np.array(values)
        dataset['word1'] = indices[:, 0]
        dataset['word2'] = indices[:, 1]
        return dataset

    def get_spearman_r(self, csls, metrics):
        metrics['crosslingual'] = {}
        if not self.flip:
            src = 'word1'
            tgt = 'word2'
        else:
            src = 'word2'
            tgt = 'word1'

        for dname in self.datasets:
            dataset = self.datasets[dname]
            e1, e2 = csls.map_to_tgt(dataset[src]), csls.tgt[dataset[tgt], ...]
            score = (e1 * e2).sum(-1)
            correlation = spearmanr(score, dataset['gold_scores']).correlation
            metrics['crosslingual'][dname] = {'correlation': correlation,
                                             'found': dataset['found'],
                                             'not_found': dataset['not_found']}
        metrics['crosslingual']['mean'] = np.array([metrics['crosslingual'][dname]['correlation'] for dname in metrics['crosslingual']]).mean()
        return metrics
