from __future__ import absolute_import
import faiss
import numpy as np
from torch.autograd import Variable
import torch
import logging

from bliss.data import Language
from bliss.utils import to_numpy, to_cuda


logger = logging.getLogger(__name__)


class CSLS(object):
    """
    Class that handles tasks related to Cross-domain Similarity Local Scaling
    """

    def __init__(self, src, tgt, map_src=None, map_tgt=None, k=10, gpu=True, gpu_device=0):
        """
        inputs:
            :param src (np.ndarray) : the source np.ndarray object
            :param tgt (np.ndarray) : the target np.ndarray object
            :param map_src (linear layer) : the Linear Layer for mapping the source (if applicable)
            :param map_tgt (linear layer) : the Linear Layer for mapping the target (if applicable)
            :param k (int) : the number of nearest neighbours to use (default: 10, as in paper)
        """
        if map_src is None:
            self.src = to_numpy(normalize(src), gpu)
        else:
            self.src = to_numpy(normalize(map_src(Variable(src))), gpu)

        if map_tgt is None:
            self.tgt = to_numpy(normalize(tgt), gpu)
        else:
            self.tgt = to_numpy(normalize(self.map_tgt(Variable(tgt))), gpu)

        self.k = k
        self.gpu = gpu
        self.gpu_device = gpu_device

        self.r_src = get_mean_similarity(self.src, self.tgt, self.k, self.gpu, self.gpu_device)
        self.r_tgt = get_mean_similarity(self.tgt, self.src, self.k, self.gpu, self.gpu_device)

    def map_to_tgt(self, source_indices):
        return self.src[source_indices, ...]

    def get_closest_csls_matches(self, source_indices, n, mode="csls"):
        """
        Gets the n closest matches of the elements located at the source indices in the target embedding.
        Returns: indices of closest matches and the mean CSLS of all these matches.
            This function maps the indices internally.
        inputs:
            :param source_indices (np.ndarray) : the source indices (in the source domain)
            :param n (int) : the number of closest matches to obtain
        """
        logger.info("Using Mode: {0}".format(mode))
        tgt_tensor = to_cuda(torch.Tensor(self.tgt), self.gpu).t()
        src_tensor = torch.Tensor(self.map_to_tgt(source_indices))

        r_src_tensor = to_cuda(torch.Tensor(self.r_src[source_indices, np.newaxis]), self.gpu)
        r_tgt_tensor = to_cuda(torch.Tensor(self.r_tgt[np.newaxis, ...]), self.gpu)

        batched_list = []
        batched_list_idx = []
        batch_size = 512
        for i in range(0, src_tensor.shape[0], batch_size):
            src_tensor_indexed = to_cuda(src_tensor[i: i + batch_size], self.gpu)
            r_src_tensor_indexed = r_src_tensor[i: i + batch_size]
            if mode == "nn":
                batch_scores = src_tensor_indexed.mm(tgt_tensor)
            elif mode == "csls":
                batch_scores = (2 * src_tensor_indexed.mm(tgt_tensor)) - r_src_tensor_indexed - r_tgt_tensor
            elif mode == "cdm":
                mu_x = torch.sqrt(1. - r_src_tensor_indexed)
                mu_y = torch.sqrt(1. - r_tgt_tensor)
                dxy = 1. - src_tensor_indexed.mm(tgt_tensor)
                eps = 1e-3
                batch_scores = -dxy / (mu_x + mu_y + eps)
            else:
                raise NotImplementedError("{0} not implemented yet".format(mode))
            best_scores, best_ix = batch_scores.topk(n)
            batched_list.append(best_scores)
            batched_list_idx.append(best_ix)
        return to_numpy(torch.cat(batched_list_idx, 0), self.gpu), to_numpy(torch.cat(batched_list, 0), self.gpu)


def get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets source points'/embeddings' nearest neighbours with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix in which nearest neighbours are to be found
        :param k (int) : the number of nearest neightbours to find
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    outputs:
        :returns distance (np.ndarray) : [len(emb_src), k] matrix of distance of
            each source point to each of its k nearest neighbours
        :returns indices (np.ndarray) : [len(emb_src), k] matrix of indices of
            each source point to each of its k nearest neighbours
    """
    if use_gpu:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = gpu_device
        index = faiss.GpuIndexFlatIP(res, emb_wrt.shape[1], cfg)
    else:
        index = faiss.IndexFlatIP(emb_wrt.shape[1])
    index.add(emb_wrt.astype('float32'))
    return index.search(emb_src.astype('float32'), k)


def get_mean_similarity(emb_src, emb_wrt, k, use_gpu=True, gpu_device=0):
    """
    Gets the mean similarity of source embeddings with respect to a set of target embeddings.
    inputs:
        :param emb_src (np.ndarray) : the source embedding matrix
        :param emb_wrt (np.ndarray) : the embedding matrix wrt which the similarity is to be calculated
        :param k (int) : the number of points to be used to find mean similarity
        :param use_gpu (bool) : true if the gpu is to be used
        :param gpu_device (int) : the GPU to be used
    """
    nn_dists, _ = get_faiss_nearest_neighbours(emb_src, emb_wrt, k, use_gpu, gpu_device)
    return nn_dists.mean(1)


def normalize(arr):
    """
    Normalizes a vector of vectors into a vector of unit vectors
    """
    return arr / torch.norm(arr, p=2, dim=1).unsqueeze(1)


def _csls_test(range_indices):
    lang = Language('en')
    lang.load('wiki.en.test.vec')
    source_word_embeddings = lang.embeddings
    dest_word_embeddings = lang.embeddings

    csls = CSLS(source_word_embeddings, dest_word_embeddings, gpu=True)
    target_indices, mean_metric = csls.get_closest_csls_matches(range_indices, 1)

    print(target_indices)
    assert (target_indices == range_indices).all()


if __name__ == "__main__":
    _csls_test(range(20, 30))
