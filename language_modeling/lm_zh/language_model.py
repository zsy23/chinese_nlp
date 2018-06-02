import os, sys
import time
import logging
import numpy as np
import tensorflow as tf

class LM(object):

    def __init__(self, dataset, config, model_dir, summary_dir):
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.config = config
        self.model_dir = model_dir
        self.summary_dir = summary_dir
        self.logger = logging.getLogger('lm_zh')
        self.algo = config.algo
        
        self.best_perplexity = 1e8

        self.iterator = tf.data.Iterator.from_structure(dataset.data_types,
                                                    dataset.data_shapes)

        if self.algo == 'lstm':
            self.build_lstm_graph()
        elif self.algo == 'gcnn':
            self.build_gcnn_graph()

        writer = tf.summary.FileWriter(self.summary_dir, tf.get_default_graph())
        writer.close()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_lstm_graph(self):
        config = self.config
        vocab_size = self.vocab.vocab_size

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                           trainable=False)

        with tf.name_scope('input'):
            self.x, self.y, self.w = self.iterator.get_next()
            # self.x = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
            # self.y = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
            # self.w = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])

            self.keep_prob = tf.get_variable('keep_prob', [], dtype=tf.float32, trainable=False)
            self.new_keep_prob = tf.placeholder(tf.float32, shape=[], name="new_keep_prob")
            self.keep_prob_update = tf.assign(self.keep_prob, self.new_keep_prob)

            self.lr = tf.get_variable('lr', [], dtype=tf.float32, trainable=False)
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self.lr_update = tf.assign(self.lr, self.new_lr)            
        
        with tf.name_scope('embedding'):
            self.embed = tf.get_variable('embedding', 
                        shape=[vocab_size, config.embed_size], 
                        dtype=tf.float32)
            self.embed_x = tf.nn.embedding_lookup(self.embed, self.x)
            
            if config.keep_prob < 1.0:
                self.embed_x = tf.nn.dropout(self.embed_x, config.keep_prob)
        
        with tf.name_scope('lstm'):
            cells = []
            for _ in range(config.num_layers):
                cell = tf.contrib.rnn.LSTMBlockCell(
                    config.hidden_size,
                    forget_bias=0.0
                )
                if config.keep_prob < 1.0:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            self.outputs, _ = tf.nn.dynamic_rnn(cell, self.embed_x, dtype=tf.float32)
            self.outputs = tf.reshape(self.outputs, [config.batch_size * config.num_steps, config.hidden_size])
        
        with tf.name_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_w', [vocab_size, config.hidden_size], dtype=tf.float32)
            self.softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)

        with tf.name_scope('loss'):
            if config.num_sampled > 0:
                labels = tf.reshape(self.y, [config.batch_size * config.num_steps, 1])
                self.loss = tf.nn.sampled_softmax_loss(
                    weights=self.softmax_w,
                    biases=self.softmax_b,
                    labels=labels,
                    inputs=self.outputs,
                    num_sampled=config.num_sampled,
                    num_classes=vocab_size,
                    partition_strategy="div")
            else:
                labels = tf.reshape(self.y, [config.batch_size * config.num_steps])
                logits = tf.matmul(self.outputs, tf.transpose(self.softmax_w))
                logits = tf.nn.bias_add(logits, self.softmax_b)
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits)
            self.loss = tf.reduce_mean(tf.reshape(self.loss, [config.num_steps, config.batch_size]) * tf.reshape(tf.to_float(self.w), [config.num_steps, config.batch_size]), axis=1)
            self.loss = tf.reshape(self.loss, [config.num_steps])
            self.loss = tf.reduce_sum(self.loss)

        with tf.name_scope('optimize'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        with tf.name_scope('ema'):
            lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*lstm.*")
            ema = tf.train.ExponentialMovingAverage(decay=0.999)
            self.train_op = tf.group(*[self.train_op, ema.apply(lstm_vars)])

    def build_gcnn_graph(self):
        config = self.config
        vocab_size = self.vocab.vocab_size

        self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                           trainable=False)

        with tf.name_scope('input'):
            self.x, self.y, self.w = self.iterator.get_next()
            # self.x = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
            # self.y = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
            # self.w = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])

            paddings = tf.constant([[0,0],[config.filter_w // 2,0]])
            self.padded_x = tf.pad(self.x, paddings, "CONSTANT")
            paddings = tf.constant([[0,0],[0,config.filter_w // 2]])
            self.padded_y = tf.pad(self.y, paddings, "CONSTANT")
            self.padded_w = tf.pad(self.w, paddings, "CONSTANT")

            self.keep_prob = tf.get_variable('keep_prob', [], dtype=tf.float32, trainable=False)
            self.new_keep_prob = tf.placeholder(tf.float32, shape=[], name="new_keep_prob")
            self.keep_prob_update = tf.assign(self.keep_prob, self.new_keep_prob)

            self.lr = tf.get_variable('lr', [], dtype=tf.float32, trainable=False)
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self.lr_update = tf.assign(self.lr, self.new_lr)       

        with tf.name_scope('embedding'):
            self.embed = tf.get_variable('embedding', 
                        shape=[vocab_size, config.embed_size], 
                        dtype=tf.float32)
            self.embed_x = tf.nn.embedding_lookup(self.embed, self.padded_x)
            
            if config.keep_prob < 1.0:
                self.embed_x = tf.nn.dropout(self.embed_x, config.keep_prob)

        with tf.name_scope('gcnn'):
            width = config.num_steps + config.filter_w // 2
            self.embed_x = tf.reshape(self.embed_x, [config.batch_size, width, config.embed_size])
            h = self.embed_x

            for i in range(config.num_layers + 1):
                fanin_depth = h.get_shape()[-1]
                filter_size = config.filter_size
                shape = (config.filter_w, fanin_depth, filter_size)
                
                with tf.variable_scope('layer_%d'%i):
                    with tf.variable_scope('linear'):
                        W = tf.get_variable('W', shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
                        b = tf.get_variable('b', filter_size, tf.float32, tf.constant_initializer(1.0))
                        conv_w = tf.add(tf.nn.conv1d(h, W, stride=1, padding='SAME'), b)
                    with tf.variable_scope('gated'):
                        W = tf.get_variable('W', shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
                        b = tf.get_variable('b', filter_size, tf.float32, tf.constant_initializer(1.0))
                        conv_v = tf.add(tf.nn.conv1d(h, W, stride=1, padding='SAME'), b)                    
                    h = conv_w * tf.sigmoid(conv_v)
                    if i == 0:
                        res_input = h
                    elif i % config.block_size == 0:
                        h += res_input
                        res_input = h
            self.outputs = tf.reshape(h, [config.batch_size * width, config.filter_size])

        with tf.name_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_w', [vocab_size, config.filter_size], dtype=tf.float32)
            self.softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)

        with tf.name_scope('loss'):
            if config.num_sampled > 0:
                labels = tf.reshape(self.padded_y, [config.batch_size * width, 1])
                self.loss = tf.nn.sampled_softmax_loss(
                    weights=self.softmax_w,
                    biases=self.softmax_b,
                    labels=labels,
                    inputs=self.outputs,
                    num_sampled=config.num_sampled,
                    num_classes=vocab_size,
                    partition_strategy="div")
            else:
                labels = tf.reshape(self.padded_y, [config.batch_size * width])
                logits = tf.matmul(self.outputs, tf.transpose(self.softmax_w))
                logits = tf.nn.bias_add(logits, self.softmax_b)
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits)
            self.loss = tf.reduce_mean(tf.reshape(self.loss, [width, config.batch_size]) * tf.reshape(tf.to_float(self.padded_w), [width, config.batch_size]), axis=1)
            self.loss = tf.reshape(self.loss, [width])
            self.loss = tf.reduce_sum(self.loss)

        with tf.name_scope('optimize'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        with tf.name_scope('ema'):
            gcnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*gcnn.*")
            ema = tf.train.ExponentialMovingAverage(decay=0.999)
            self.train_op = tf.group(*[self.train_op, ema.apply(gcnn_vars)])

    def train(self):
        config = self.config
        dataset = self.dataset

        train_init = self.iterator.make_initializer(dataset.train_data)
        valid_init = self.iterator.make_initializer(dataset.valid_data)

        for i in range(config.epoch):
            self.logger.info('Start epoch {} ...'.format(i + 1))
            print('Epoch %d' % (i + 1, ))
            total_loss = 0
            total_steps = 0
            start_time = time.time()

            self.sess.run(train_init)
            lr_decay = config.lr_decay ** max(i + 1 - config.lr_keep_epoch, 0.0)
            self.sess.run(self.lr_update, feed_dict={self.new_lr: config.lr * lr_decay})
            self.sess.run(self.keep_prob_update, feed_dict={self.new_keep_prob: config.keep_prob})
            try:
                while True:
                    _, loss = self.sess.run([self.train_op, self.loss])

                    total_loss += loss
                    total_steps += config.num_steps

                    sys.stdout.write('Process: %.3f, perplexity: %.3f, speed: %.1fk wps\r' %
                            (total_steps * config.batch_size * 1.0 / dataset.train_size, 
                            np.exp(total_loss / total_steps),
                            total_steps * config.batch_size / (time.time() - start_time) / 1000.0 ))
                    sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                sys.stdout.write('\n')
                sys.stdout.flush()
                self.logger.info('Epoch %d perplexity: %.3f' % (i + 1, np.exp(total_loss / total_steps)))

            self.logger.info('Start eval on valid data ...')
            valid_loss = 0
            valid_steps = 0
            self.sess.run(valid_init)
            self.sess.run(self.keep_prob_update, feed_dict={self.new_keep_prob: 1.0})
            try:
                while True:
                    loss = self.sess.run(self.loss)

                    valid_loss += loss
                    valid_steps += config.num_steps

            except tf.errors.OutOfRangeError:
                pass
            self.logger.info('Valid data perplexity: %.3f' % (np.exp(valid_loss / valid_steps), ))
            print('Valid data perplexity: %.3f' % (np.exp(valid_loss / valid_steps), ))

            if np.exp(valid_loss / valid_steps) < self.best_perplexity:
                self.best_perplexity = np.exp(valid_loss / valid_steps)
                self.save()
        
    def test(self):
        config = self.config
        dataset = self.dataset

        test_init = self.iterator.make_initializer(dataset.test_data)

        total_loss = 0
        total_steps = 0

        self.logger.info('Start eval on test data ...')
        self.sess.run(test_init)
        self.sess.run(self.keep_prob_update, feed_dict={self.new_keep_prob: 1.0})
        try:
            while True:
                loss = self.sess.run(self.loss)

                total_loss += loss
                total_steps += config.num_steps

        except tf.errors.OutOfRangeError:
            pass
        self.logger.info('Test data perplexity: %.3f' % (np.exp(total_loss / total_steps), ))
        print('Test data perplexity: %.3f' % (np.exp(total_loss / total_steps), ))

    def save(self):
        self.logger.info('Save model to {}, with prefix {} ...'.format(self.model_dir, self.algo))
        self.saver.save(self.sess, os.path.join(self.model_dir, self.algo))

    def restore(self):
        self.logger.info('Model restore from {}, with prefix {} ...'.format(self.model_dir, self.algo))
        self.saver.restore(self.sess, os.path.join(self.model_dir, self.algo))





        
