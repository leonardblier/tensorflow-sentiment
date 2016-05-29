from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader
import imdb_data


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

class SentimentModel(object):
    """The sentiment model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        size = config.hidden_size
        self.max_len = max_len = config.max_len
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, config.max_len])
        self._targets = tf.placeholder(tf.int32, [batch_size])

        embedding = tf.get_variable("embedding", [vocab_size, size])
        inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        output = tf.reduce_sum(inputs, 1)
        softmax_w = tf.get_variable("softmax_w", [size, 2])
        softmax_b = tf.get_variable("softmax_b", [2])
        
        logits = tf.matmul(output, softmax_w) + softmax_b
        prediction = tf.nn.softmax(logits)
        self._prediction = prediction

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._targets)
        
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
        
    @property
    def input_data(self):
        return self._input_data
    
    @property
    def targets(self):
        return self._targets

    @property
    def prediction(self):
        return self._prediction
    
    
    @property
    def cost(self):
        return self._cost
    
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op


    
class Config(object):
    init_scale = 0.1
    learning_rate = 1.
    max_grad_norm = 5
    hidden_size = 20
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.1
    batch_size = 1
    vocab_size = 100000
    max_len = 100




def run_epoch(session, m, data, eval_op, id2word, verbose=False):
    """Runs the model on the given data."""
    epoch_size = len(data[0])//m.batch_size
    start_time = time.time()
    costs = 0.0
    correct_answers = 0.0

    seqs, labels = data
    MAXLEN = 100
    for step in range(epoch_size):
        x = seqs[step*m.batch_size:(step+1)*m.batch_size]
        y = labels[step*m.batch_size:(step+1)*m.batch_size]
        x, max_len_seqs, y = imdb_data.prepare_data(x, y, MAXLEN)
        x = x[:,:MAXLEN]
        cost, prediction, _ = session.run([m.cost, m.prediction, eval_op],
                                     {m.input_data: x,
                                      m.targets: y})
        correct_answers += (np.argmax(prediction, 1) == np.array(y)).sum()
        costs += cost

        if verbose and step % (epoch_size // 10) == 10:
            print("Sentence : "+imdb_data.seq2str(x[0],id2word))
            print("True label : "+str(y[0]))
            print("Predicted label : "+str(np.argmax(prediction[0])))
            print("%.3f loss: %.3f accuracy: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, costs / step,
                   correct_answers / (step*m.batch_size),
                   step * m.batch_size / (time.time() - start_time)))

    return (costs / epoch_size, correct_answers / (epoch_size * m.batch_size))



def main(_):
    if not FLAGS.data_path:
        #raise ValueError("Must set --data_path to PTB data directory")
        pass

    train_data, valid_data, test_data = imdb_data.load_data()
    word2id, id2word = imdb_data.load_dict_imdb()
    

    config = Config()
    eval_config = Config()
    eval_config.batch_size = 1
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = SentimentModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = SentimentModel(is_training=False, config=config)
            mtest = SentimentModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        print("Starting")
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity, train_accuracy = run_epoch(session, m, train_data, m.train_op, id2word,
                                       verbose=True)
            print("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % \
                  (i + 1, train_perplexity, train_accuracy))
            valid_perplexity, valid_accuracy = run_epoch(session,
                                                         mvalid,
                                                         valid_data,
                                                         tf.no_op(),
                                                         id2word)
            print("Epoch: %d Valid Perplexity: %.3f Valid Accuracy: %.3f" % \
                  (i + 1, valid_perplexity, valid_accuracy))

        test_perplexity, test_accuracy = run_epoch(session,
                                                   mtest,
                                                   test_data,
                                                   tf.no_op(),
                                                   id2word)
        print("Test Perplexity: %.3f Test Accuracy: %.3f" % (test_perplexity,test_accuracy))


if __name__ == "__main__":
    tf.app.run()
