'''
Util functions providing logics for extracting global rationales from a dataset.
The input to all functions is a list of lists of importance scores (as numpy arrays), the output is arguments for extraction from each.
These can be input in a module implementing RationaleExtractor, for extract_rationale().
'''
import math
import numpy as np


def global_argsort(arr):
    '''
    :returns: tupled array of ordered indices
    '''
    n,l = arr.shape
    return [(i//l,i%l) for i in arr.reshape(1,-1).argsort()][0]


def max_unconstrained(weights, lengths, max_ratio):
    max_tokens = math.ceil(sum(lengths) * max_ratio)
    glob_sort = global_argsort(weights)
    return tuple([g[-max_tokens:] for g in glob_sort])  # indexes into sentence tokens


def max_limited_min(weights, lengths, max_ratio, min_inst_ratio):
    '''
    first fill up on min_inst_ratio from each instance, then add the rest
    :param min_inst_ratio: threshold of minimal words to extract from each instance
    '''
    n = weights.shape[0]
    glob_sort = global_argsort(weights)
    rev_glob_sort = reversed([(glob_sort[0][i], glob_sort[1][i]) for i in range(len(glob_sort[0]))])
    remaining = [math.ceil(l * min_inst_ratio) for l in lengths]
    max_tokens = math.ceil(sum(lengths) * max_ratio)
    buff = []  # buffer for at-threshold instance tokens
    tok_idcs = []  # return indices
    while max(remaining) > 0 and len(tok_idcs) < max_tokens:
        for tup in rev_glob_sort:
            n, t = tup
            if remaining[n] <= 0:
                buff.append(tup)
            else:
                tok_idcs.append(tup)
                remaining[n] -= 1
    tok_idcs.extend(buff)

    return tok_idcs[:max_tokens]


def max_limited_min_trunc(weights, lengths, max_ratio, min_inst_ratio, top_k):
    '''
    :param top_k: leave only up to this many words from each instance
    '''
    trunc_weights = np.array([trunc_arr(a, top_k) for a in weights])
    trunc_lengths = [min(l, top_k) for l in lengths]
    return max_limited_min(trunc_weights, trunc_lengths, max_ratio, min_inst_ratio)


def trunc_arr(a, k):
    kth_val = np.sort(a)[-k]
    b = a * (a > kth_val)
    return b / np.linalg.norm(b, 1)


