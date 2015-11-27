"""
Reads twitter data, cleans it and presents it as one big sequence, ready
for the model to try and learn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import collections
import os
import sys
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange
import tensorflow as tf
import logging
logger = logging.getLogger('charmodel.reader')

# the default set of chars
BOOTSTRAP_PATH = os.path.abspath('./bootstrap_data.txt')
DEFAULT_CHARS = {}

def _read_chars(filename):
    """reads a file into a big long list"""
    #with gfile.GFile(filename, "r") as f:
    #    return list(f.read())
    import codecs
    with codecs.open("bootstrap_data.txt", "r", "utf-8", errors="ignore") as f:
        return list(f.read())

def _build_vocab(filename):
    data = _read_chars(filename) + [u'\ufffd'] # jic
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    # yeah I nicked this from the tf penn treebank example
    # I don't really understand it
    word_to_id = dict(zip(words,range(len(words))))
    return word_to_id

def _clean_chars(chars, allowed, replace_with=u'\ufffd'):
    """Cleans a sequence of chars, replacing those not in the
    allowed sequence with the specified replacement character"""
    return [c if c in allowed else replace_with for c in chars]

def _unique_chars(chars):
    return dict([(c,c) for c in chars])

def data_to_ids(data, clean=True, allowed=DEFAULT_CHARS):
    """Returns a list of integer ids and the vocabulary used to generate it"""
    # this one is the main one that people will use
    if clean:
        data = _clean_chars(data, allowed)
    return [VOCAB[c] for c in data]
    
def get_vocab():
    return VOCAB

def get_bootstrap():
    return data_to_ids(_read_chars(BOOTSTRAP_PATH), clean=False)

def tweets_to_sequence(data, append_bootstrap=True):
    """Ok, maybe this one is more likely"""
    # we don't need any kind of test or validation data
    ids = data_to_ids(data)
    if append_bootstrap:
        return ids + get_bootstrap()
    return ids

def tweet_iterator(raw_data, batch_size, num_steps):
    """Iterate the data.
    Generates batch_size sequence pairs.

    Yields:
        pairs of batched data, the second element is the first
        shifter to the right by one.
        Hence we can use it to learn to generate sequences.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i+1)]
    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch size == 0, decrease batch size or num steps")
    
    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x,y)


if not os.path.exists(BOOTSTRAP_PATH):
    logger.error("Can't find bootstrap data, no default set of characters! Cleaning may prove difficult.")
    logger.error("Boostrap filepath {}".format(BOOTSTRAP_PATH))
    DEFAULT_CHARS={}
    logger.error("also no vocab")
else:
    dchars = _read_chars(BOOTSTRAP_PATH)
    DEFAULT_CHARS = _unique_chars(dchars)
    VOCAB = _build_vocab(BOOTSTRAP_PATH)

