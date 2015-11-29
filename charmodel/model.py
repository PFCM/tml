"""
This is some lstm business, ie. the fun part.
Goal is to make sure that you can load a model
from a file, train it a bit and generate a 
sample.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import time
import logging

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

from .reader import *

class Config(object):
    """it's nice to have these things in one place"""
    def __init__(self):
        self.init_scale = 0.1 # only matters the first time
        self.learning_rate = 1.0 # probably want to turn it down subsequently
        self.max_grad_norm = 5 # seems ok
        self.num_layers = 2 # did ok before
        self.hidden_size = 128 # not a bad tradeoff
        self.max_epoch = 1 # when to decay the learning rate
        self.max_max_epoch = 10 # how many actually
        self.keep_prob = 0.5 # for the dropout
        self.lr_decay = 0.8 # helps
        self.batch_size = 32 # tuning performance
        self.vocab_size = 154 # will get set anyway
        self.num_steps = 140

class CharModel(object):
    """The actual model, handles putting together the graph"""
    def __init__(self, is_training, config):
        """constructs a graph"""
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps],
                                          name="input_data")
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps],
                                       name="targets")

        # here it is
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # do an embedding (always on cpu)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.split(
                1, num_steps, tf.nn.embedding_lookup(embedding, self._input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        if is_training and config.keep_prob < 1:
            inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

        from tensorflow.models.rnn import rnn
        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        
        # reshape
        outputs = tf.reshape(tf.concat(1, outputs), [-1, size])
        
        logits = tf.nn.xw_plus_b(outputs,
                                 tf.get_variable("softmax_W", [size,vocab_size]),
                                 tf.get_variable("softmax_b", [vocab_size]))
        self._softmax_out = tf.nn.softmax(logits) # this is just used for sampling
        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self._targets,[-1])],
                                                [tf.ones([batch_size * num_steps])],
                                                vocab_size)
        self._cost = cost = tf.div(tf.reduce_sum(loss),
                                   tf.constant(batch_size, dtype=tf.float32))
        self._final_state = states[-1]

        if not is_training:
            return # don't need to optimisation ops

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        # actually the simple guy does good
        # with the grad clipping and the lr schedule and whatnot
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def softmax_out(self):
        return self._softmax_out

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

def _run_epoch(session, m, data, eval_op, verbose=False):
    """run one of model on some of data"""
    epoch_size = ((len(data) // m.batch_size) -1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    # here we iter the data
    for step, (x,y) in enumerate(tweet_iterator(data,
                                                m.batch_size,
                                                m.num_steps)):
        feed_dict = {m.input_data:x,
                     m.targets:y,
                     m.initial_state: state}
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     feed_dict=feed_dict)
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            logging.info("{:3f} perplexity: {:.3f} speed: {:.0f} wps".format(
                    step*1.0 / epoch_size, np.exp(costs / iters),
                    iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs/iters)

def get_config():
    return Config()

def _sample_one(session, model, state, prev, vocab_size):
    probs, state = session.run([model.softmax_out, model.final_state],
                               {model.input_data: np.array([prev]).reshape((1,1)),
                                model.initial_state: state})
    return np.random.choice(vocab_size, p=probs.ravel()), state

def sample_sequence(session, model, length, index_to_word, start=0, init_state=None):
    if init_state:
        state = init_state.eval()
    else:
        state = model.initial_state.eval()

    seq = []
    idx = start
    for _ in range(length):
        idx, state = _sample_one(session, model, state, idx, len(index_to_word))
        seq.append(index_to_word[idx])
#    print(seq)
    return u"".join(seq)

def train_and_sample(data, config, model_dir,  vocab, sample_length=140):
    """Does a number of things.
    First builds a model.
    Then tries to load saved weights from file.
    Then trains for a while on the data.
    Then generates a sample sequence.
    Then saves the model to file."""
    
    # irl, first we need to copy the config
    from copy import copy
    sample_conf = copy(config)
    sample_conf.batch_size = 1
    sample_conf.num_steps = 1
    sample_conf.vocab_size = config.vocab_size = len(vocab)
    
    # then we need to flip the vocab dict
    idx2wrd = dict([(y,x) for x,y in vocab.iteritems()])

    # now we build
    with tf.Graph().as_default(),tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                     config.init_scale)
        logging.info("Building models")
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = CharModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            msample = CharModel(is_training=False, config=sample_conf)

        save = tf.train.Saver() # default is save the lot
        # now let's see if we can restore
        done_init = False
        if os.path.exists(model_dir):
            if os.path.isfile(model_dir):
                model_dir = os.path.dirname(model_dir)
            model_file = os.path.join(model_dir,
                                      "{}x{}.chkpt".format(config.hidden_size,
                                                           config.num_layers))
            if os.path.exists(model_file):
                logging.info("found latest checkpoint: {}".format(model_file))
                save.restore(session, model_file)
                done_init = True
                logging.info("succesfully initialised from file")
            else:
                model_file = model_dir + "{}x{}.chkpt".format(config.hidden_size,
                                                              config.num_layers)
        else:
            model_file = model_dir + "{}x{}.chkpt".format(config.hidden_size,
                                                          config.num_layers)

        if not done_init:
            logging.info("initialising weights randomly")
            tf.initialize_all_variables().run()

        logging.info("About to begin training")
        try:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i-config.max_epoch, 0.0)
                mtrain.assign_lr(session, config.learning_rate*lr_decay)
                logging.info("Epoch {} learning rate: {:.3f}".format(
                        i+1,
                        session.run(mtrain.lr)))
                logging.info(u"sneak peek: {}".format(sample_sequence(
                            session, msample, 100, idx2wrd, start=vocab[u'\n'])))
                
                train_perplexity = _run_epoch(session,
                                              mtrain,
                                              data,
                                              mtrain.train_op,
                                              verbose=True)
                logging.info("{}    train perplexity: {:.3f}".format(
                        i+1, train_perplexity))
                        
                if (i+1) % 2 == 0 or i == config.max_max_epoch-1:
                    path = save.save(session, model_file)
                    logging.info("{}    saved model at: {}".format(i+1,path))
        except KeyboardInterrupt as e: # and anything else?
            logging.info("Training ended prematurely: {}".format(e))
        # NOTE choose a random start position
        sample = sample_sequence(session, msample, sample_length, idx2wrd, start=vocab[u'A'])
        logging.info(u"Final sample: {}".format(sample))
        save.save(session, model_file)
        return sample
                                 
