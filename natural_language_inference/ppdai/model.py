import os
import numpy as np
import tensorflow as tf

import data_utils
import misc_utils as utils


class Model(object):

    def __init__(self, hparams, mat, iterator, mode):
        self.hparams = hparams
        self.mat = mat
        self.iterator = iterator
        self.mode = mode

        self.dropout = hparams.dropout
        self.hidden_size = hparams.hidden_size

        # Initializer
        initializer = self.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        self.build_graph()

        self.saver = tf.train.Saver()


    def get_initializer(self, init_op, seed=None, init_weight=None):
        if init_op == "uniform":
            assert init_weight
            return tf.random_uniform_initializer(
                -init_weight, init_weight, seed=seed)
        elif init_op == "glorot_normal":
            return tf.glorot_normal_initializer(
                seed=seed)
        elif init_op == "glorot_uniform":
            return tf.glorot_uniform_initializer(
                seed=seed)
        else:
            raise ValueError("Unknown init_op %s" % init_op)


    def create_rnn_cell(self, num_layers, hidden_size):
        cell_list = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hidden_size)

            cell_list.append(cell)

        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)

        return cell
    

    def mask_3d(self, values, sentence_sizes, mask_value, dimension=2):

        if dimension == 1:
            values = tf.transpose(values, [0, 2, 1])
        time_steps1 = tf.shape(values)[1]
        time_steps2 = tf.shape(values)[2]

        ones = tf.ones_like(values, dtype=tf.float32)
        pad_values = mask_value * ones
        mask = tf.sequence_mask(sentence_sizes, time_steps2)

        # mask is (batch_size, sentence2_size). we have to tile it for 3d
        mask3d = tf.expand_dims(mask, 1)
        mask3d = tf.tile(mask3d, (1, time_steps1, 1))

        masked = tf.where(mask3d, values, pad_values)

        if dimension == 1:
            masked = tf.transpose(masked, [0, 2, 1])

        return masked


    def build_esim(self):
        hidden_size = self.hidden_size
        q1_embed = self.q1_embed
        q1_len = self.q1_len
        q2_embed = self.q2_embed
        q2_len = self.q2_len

        with tf.variable_scope('esim'):
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                q1_embed = tf.nn.dropout(q1_embed, (1.0 - self.dropout))
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                q2_embed = tf.nn.dropout(q2_embed, (1.0 - self.dropout))

            fw_w_cell = self.create_rnn_cell(1, hidden_size)
            bw_w_cell = self.create_rnn_cell(1, hidden_size)

            q1_bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_w_cell,
                bw_w_cell,
                q1_embed,
                sequence_length=q1_len,
                dtype=tf.float32)

            q2_bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_w_cell,
                bw_w_cell,
                q2_embed,
                sequence_length=q2_len,
                dtype=tf.float32)

            q1_bi_rep = tf.concat(q1_bi_outputs, -1)
            q2_bi_rep = tf.concat(q2_bi_outputs, -1)

            with tf.variable_scope('attend'):
                attentions = tf.matmul(q1_bi_rep, tf.transpose(q2_bi_rep, [0, 2, 1]))

                masked1 = self.mask_3d(attentions, q2_len, -np.inf)
                q1_attn = tf.nn.softmax(masked1, axis=2)
                masked2 = self.mask_3d(tf.transpose(attentions, [0, 2, 1]), q1_len, -np.inf)
                q2_attn = tf.nn.softmax(masked2, axis=2)

                alpha = tf.matmul(q2_attn, q1_bi_rep)
                beta = tf.matmul(q1_attn, q2_bi_rep)

            with tf.variable_scope('compare'):
                inputs = tf.concat(
                    [q1_bi_rep, beta, 
                    q1_bi_rep * beta, q1_bi_rep - beta],
                    -1)
                
                v1 = tf.layers.dense(inputs, hidden_size, tf.nn.relu, name='combine')
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                    v1 = tf.nn.dropout(v1, (1.0 - self.dropout))
                
                inputs = tf.concat(
                    [q2_bi_rep, alpha,
                    q2_bi_rep * alpha, q2_bi_rep - alpha],
                    -1)

                v2 = tf.layers.dense(inputs, hidden_size, tf.nn.relu, name='combine', reuse=True)
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                    v2 = tf.nn.dropout(v2, (1.0 - self.dropout))

            with tf.variable_scope('aggregate'):
                fw_w_cell = self.create_rnn_cell(1, hidden_size)
                bw_w_cell = self.create_rnn_cell(1, hidden_size)

                q1_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_w_cell,
                    bw_w_cell,
                    v1,
                    sequence_length=q1_len,
                    dtype=tf.float32)

                q2_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_w_cell,
                    bw_w_cell,
                    v2,
                    sequence_length=q2_len,
                    dtype=tf.float32)

                q1 = tf.concat(q1_outputs, -1)
                q2 = tf.concat(q2_outputs, -1)
                q1 = self.mask_3d(q1, self.q1_len, 0, 1)
                q2 = self.mask_3d(q2, self.q2_len, 0, 1)

                v1_avg = tf.reduce_sum(q1, 1) / tf.expand_dims(tf.to_float(q1_len), 1)
                v2_avg = tf.reduce_sum(q2, 1) / tf.expand_dims(tf.to_float(q2_len), 1)
                v1_max = tf.reduce_max(q1, 1)
                v2_max = tf.reduce_max(q2, 1)
                v = tf.concat([v1_avg, v1_max, v2_avg, v2_max], -1)

                if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                    v = tf.nn.dropout(v, (1.0 - self.dropout))
                v = tf.layers.dense(v, hidden_size, tf.nn.tanh)
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN and self.dropout > 0.0:
                    v = tf.nn.dropout(v, (1.0 - self.dropout))
                logits = tf.layers.dense(v, 2)

                return logits
                

    def build_graph(self):
        max_grad_norm = self.hparams.max_grad_norm

        self.global_step = tf.Variable(0, trainable=False)

        (self.labels, self.q1, self.q2, self.q1_len, self.q2_len) = (
            self.iterator.labels, 
            self.iterator.q1, self.iterator.q2,
            self.iterator.q1_len, self.iterator.q2_len)

        self.batch_size = tf.size(self.labels)

        self.lr = tf.placeholder(tf.float32, name='lr')

        with tf.variable_scope('embedding'):
            self.embed = tf.get_variable(
                'embedding',
                initializer=tf.constant(self.mat, dtype=tf.float32),
                trainable=True)

            self.q1_embed = tf.nn.embedding_lookup(
                self.embed,
                self.q1
            )

            self.q2_embed = tf.nn.embedding_lookup(
                self.embed,
                self.q2
            )

        self.logits = self.build_esim()
        self.scores = tf.nn.softmax(self.logits, -1)
        
        params = tf.trainable_variables()

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.variable_scope('optimizer'):
                opt = tf.train.AdamOptimizer(self.lr)

                gradients = tf.gradients(
                    self.loss,
                    params)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, max_grad_norm)

                self.update = opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)

        utils.log("Trainable variables")
        for param in params:
            utils.log("%s, %s" % (param.name, str(param.get_shape())))
            
            
    def train(self, sess, lr):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run(
                    [self.global_step, 
                    self.batch_size,
                    self.loss, 
                    self.update], feed_dict={self.lr: lr})


    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run(
                    [self.batch_size, self.loss])


    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run(self.scores)


    def save(self, sess, model_dir, model_prefix):
        self.saver.save(sess, os.path.join(model_dir, model_prefix))
        utils.log('Model saved in {} with prefix {}'.format(model_dir, model_prefix))

    
    def restore(self, sess, model_dir, model_prefix):
        self.saver.restore(sess, os.path.join(model_dir, model_prefix))
        utils.log('Model restored from {} with prefix {}'.format(model_dir, model_prefix))
