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


def padded_collate(vector[vector[vector[int]]] examples):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    cdef vector[vector[int]] source, target
    cdef vector[int] utr
    cdef Py_ssize_t btc_idx, btc_size = examples.size()
    cdef int utr_len, src_max_len = 0, trg_max_len = 0

    for btc_idx in range(btc_size):
        # source size length
        utr = examples[btc_idx][0]
        utr_len = utr.size()
        if utr_len > src_max_len:
            src_max_len = utr_len

        source.push_back(utr)

        # target size length
        utr = examples[btc_idx][1]
        utr_len = utr.size()
        if utr_len > trg_max_len:
            trg_max_len = utr_len

        target.push_back(utr)

    cdef np.ndarray[np.int32_t, ndim=3] source_ids = \
        batchify(source, src_max_len)

    cdef np.ndarray[np.int32_t, ndim=3] target_ids = \
        batchify(source, src_max_len)

    return source_ids[0], source_ids[1], target_ids[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef batchify(vector[vector[int]] ids, const int max_len):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr

    cdef np.ndarray[np.int32_t, ndim=3] tensor = \
        np.ones([2, btc_size, max_len], dtype=np.int32)

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    for btc_idx in range(btc_size):
        utr_size = ids[btc_idx].size()
        diff_size = max_len - utr_size

        for idx in range(utr_size):
            tensor[0, btc_idx, idx] = \
                ids[btc_idx][idx]

        for diff_idx in range(diff_size):
            pad_idx = utr_size + diff_idx 
            # 0 is the hard coded pad idx
            tensor[0, btc_idx, pad_idx] = 0
            tensor[1, btc_idx, pad_idx] = 0

    return tensor
