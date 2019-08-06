"""
@author:    Patrik Purgai
@copyright: Copyright 2019, sentence-similarity
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.06.25.
"""

# distutils: language=c++

cimport numpy as np
import numpy as np

import cython

from libcpp.vector cimport vector
from libcpp cimport bool


def padded_collate(
        vector[vector[vector[int]]] examples):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    # unzipping the examples lists
    cdef vector[vector[int]] utterances
    cdef vector[int] labels
    cdef Py_ssize_t btc_size = examples.size()
    cdef Py_ssize_t btc_idx

    # creating variable for storing the list of
    # sampled utterances `utr_smp`
    cdef vector[vector[int]] utr_smp
    cdef vector[int] utr
    cdef int num_smp

    cdef utr_len, max_len = 0

    for btc_idx in range(btc_size):
        utr_smp = examples[btc_idx]
        utr = utr_smp[0]
        label = utr_smp[1][0]

        utr_len = utr.size()
        if utr_len > max_len:
            max_len = utr_len

        utterances.push_back(utr)
        labels.push_back(label)

    cdef Py_ssize_t btc_size = examples.size()

    cdef np.ndarray[np.int32_t, ndim=3] inputs = \
        np.empty([btc_size, max_len, 2], dtype=np.int32)

    cdef Py_ssize_t utr_size, tok_idx, diff_size, \
        diff_idx, pad_idx

    for btc_idx in range(btc_size):
        utr_size = examples[btc_idx].size()
        diff_size = max_len - utr_size
        for tok_idx in range(utr_size):
            inputs[btc_idx, tok_idx, 0] = \
                examples[btc_idx][tok_idx]
            inputs[btc_idx, tok_idx, 1] = 1
        for diff_idx in range(diff_size):
            pad_idx = utr_size + diff_idx 
            # 0 is the hard coded pad idx
            inputs[btc_idx, pad_idx, 0] = 0
            inputs[btc_idx, pad_idx, 1] = 0
    
    cdef np.ndarray[np.int32_t, ndim=2] input_ids = \
        inputs[:, :, 0]

    cdef np.ndarray[np.int32_t, ndim=2] attn_mask = \
        inputs[:, :, 1]

    return input_ids, attn_mask, labels
