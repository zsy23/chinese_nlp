# -*- coding:utf8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import json
import time, math, re
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers.pointer_net import summ, pointer
from mrc_eval import calc_metrics, normalize
from bleu import BLEUWithBonus
from rouge import RougeLWithBonus
from hack_yesno import gen_yesno


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.char_hidden_size = args.char_hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.pr_rate = args.pr_rate
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        self.max_w_len = args.max_w_len

        # the vocab
        self.vocab = vocab
        self.max_rouge_l = 0

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._passage_rank()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """

        self.word_embed = tf.get_variable("word_embed", 
                                        initializer=tf.constant(self.vocab.word_embeddings, 
                                                                dtype=tf.float32), 
                                        trainable=False)
        self.char_embed = tf.get_variable("char_embed", 
                                        initializer=tf.constant(self.vocab.char_embeddings, 
                                                                dtype=tf.float32), 
                                        trainable=False)
        self.pos_embed = tf.get_variable('pos_embed',
                        initializer=tf.constant(self.vocab.pos_embeddings,
                                                dtype=tf.float32),
                        trainable=False
        )

        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.pc = tf.placeholder(tf.int32, [None, None, None])
        self.qc = tf.placeholder(tf.int32, [None, None, None])
        self.p_em = tf.placeholder(tf.int32, [None, None])
        self.p_pos = tf.placeholder(tf.int32, [None, None])
        self.q_pos = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.pr = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.pc_length = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.pc, tf.bool), tf.int32), axis=2), [-1])
        self.qc_length = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qc, tf.bool), tf.int32), axis=2), [-1])

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        batch_size = tf.shape(self.p)[0]
        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                pc_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_embed, self.pc), 
                    [batch_size * self.max_p_len, self.max_w_len, self.vocab.char_embed_dim])
                qc_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_embed, self.qc), 
                    [batch_size * self.max_q_len, self.max_w_len, self.vocab.char_embed_dim])
                cell_fw = tf.contrib.rnn.GRUCell(self.char_hidden_size)
                cell_bw = tf.contrib.rnn.GRUCell(self.char_hidden_size)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, pc_emb, self.pc_length, dtype=tf.float32)
                pc_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qc_emb, self.qc_length, dtype=tf.float32)
                qc_emb = tf.concat([state_fw, state_bw], axis=1)
                pc_emb = tf.reshape(pc_emb, [batch_size, self.max_p_len, 2 * self.char_hidden_size])
                qc_emb = tf.reshape(qc_emb, [batch_size, self.max_q_len, 2 * self.char_hidden_size])

            with tf.name_scope("word"):
                p_emb = tf.nn.embedding_lookup(self.word_embed, self.p)
                q_emb = tf.nn.embedding_lookup(self.word_embed, self.q)

            with tf.name_scope("pos"):
                p_pos_emb = tf.nn.embedding_lookup(self.pos_embed, self.p_pos)
                q_pos_emb = tf.nn.embedding_lookup(self.pos_embed, self.q_pos)
            
            with tf.name_scope("em"):
                sh = tf.shape(self.p_em)
                resh = [sh[0], sh[1], 1]
                p_em_feat = tf.reshape(tf.cast(self.p_em, dtype=tf.float32), shape=resh)

            self.p_emb = tf.concat([p_emb, pc_emb, p_pos_emb, p_em_feat], axis=2)
            self.q_emb = tf.concat([q_emb, qc_emb, q_pos_emb], axis=2)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _passage_rank(self):
        """
        Passage Rank
        """
        with tf.variable_scope('passage_rank'):
            init = summ(self.sep_q_encodes, self.hidden_size)
            r_P, _ = pointer(self.fuse_p_encodes, init, self.hidden_size, "pr_pointer")
            concatenate = tf.concat([init,r_P],axis=1)
            tmp = tc.layers.fully_connected(concatenate, num_outputs=self.hidden_size, activation_fn=tf.nn.tanh)
            tmp2 = tc.layers.fully_connected(tmp, num_outputs=1, activation_fn=None)
            batch_size = tf.shape(self.start_label)[0]
            p_num = tf.shape(self.fuse_p_encodes)[0] / batch_size
            self.rank_p = tf.nn.softmax(tf.reshape(tmp2, [batch_size, p_num]))


    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss_1 = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss_1 += self.weight_decay * l2_loss
        
        labels = tf.one_hot(self.pr, tf.shape(self.rank_p)[1], axis=1)
        every_loss_pr = -tf.reduce_sum(labels * tf.log(self.rank_p + 1e-9), 1)
        self.loss_pr = tf.reduce_mean(every_loss_pr)
        self.loss = (1.0 - self.pr_rate) * self.loss_1 + self.pr_rate * self.loss_pr


    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob, data, batch_size, save_dir, save_prefix):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        eval_every_n_batch = (len(data.train_set) - 1) / (8 * batch_size)
        for bitx, batch in enumerate(train_batches, 1):          
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.pc: batch['passage_char_ids'],
                         self.qc: batch['question_char_ids'],
                         self.p_em: batch['passage_em'],
                         self.p_pos: batch['passage_pos'],
                         self.q_pos: batch['question_pos'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.pr: batch['passage_rank'],
                         self.dropout_keep_prob: dropout_keep_prob}

            _, loss = self.sess.run([self.train_op, self.loss], 
                                            feed_dict=feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
            
            if eval_every_n_batch > 0 and bitx % eval_every_n_batch == 0:
                self.logger.info('Evaluating the model ...')
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['ROUGE-L'] > self.max_rouge_l:
                        self.save(save_dir, save_prefix)
                        self.max_rouge_l = bleu_rouge['ROUGE-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')

        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob,
                                data, batch_size, save_dir, save_prefix)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['ROUGE-L'] > self.max_rouge_l:
                        self.save(save_dir, save_prefix)
                        self.max_rouge_l = bleu_rouge['ROUGE-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False, hack=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        print('hack', hack)

        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.pc: batch['passage_char_ids'],
                         self.qc: batch['question_char_ids'],
                         self.p_em: batch['passage_em'],
                         self.p_pos: batch['passage_pos'],
                         self.q_pos: batch['question_pos'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.pr: batch['passage_rank'],
                         self.dropout_keep_prob: 1.0}
            start_probs, end_probs, loss = self.sess.run([self.start_probs, self.end_probs, self.loss], 
                                                        feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len, hack=False)
                best_answer = best_answer.replace(',', '，')
                best_answer = best_answer.replace(';', '；')
                #best_answer = best_answer.replace('!', '！')
                best_answer = best_answer.replace(':', '：')
                #   best_answer = best_answer.replace('?', '？')
                best_answer = best_answer.replace('(', '（')
                best_answer = best_answer.replace(')', '）')
                best_answer = best_answer.replace('步骤阅读', '')
                if len(best_answer) > 0 and (best_answer[0] == '。' or best_answer[0] == '、'):
                    best_answer = best_answer[1:]
                if len(best_answer) > 0 and best_answer[-1] != '。':
                    best_answer += '。'

                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    if hack and sample['question_type'] == 'YES_NO':
                        yesno = gen_yesno(sample, best_answer)
                        if 'answers' in sample:
                            pred_answers.append({'yesno_answers': [yesno],
                                                 'question': sample['question'],
                                                 'question_type': sample['question_type'],
                                                 'answers': [best_answer],
                                                 'answers_true': sample['answers'],
                                                 'question_id': sample['question_id']
                                                })
                        else:
                            pred_answers.append({'yesno_answers': [yesno],
                                                 'question': sample['question'],
                                                 'question_type': sample['question_type'],
                                                 'answers': [best_answer],
                                                 'question_id': sample['question_id']
                                                })
                    else:
                        if 'answers' in sample:
                            pred_answers.append({'yesno_answers': [],
                                                 'question': sample['question'],
                                                 'question_type': sample['question_type'],
                                                 'answers': [best_answer],
                                                 'answers_true': sample['answers'],
                                                 'question_id': sample['question_id']
                                                })
                        else:
                            pred_answers.append({'yesno_answers': [],
                                                'question': sample['question'],
                                                'question_type': sample['question_type'],
                                                'answers': [best_answer],
                                                'question_id': sample['question_id']
                                                })

                if 'answers' in sample:
                    ref_answers.append({'entity_answers': sample.get('entity_answers', [[]]),
                                        'yesno_answers': sample.get('yesno_answers', []),
                                        'question': sample['question'],
                                        'source': 'both',
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'question_id': sample['question_id'],
                                        })

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            bleu_rouge = {}
            bleu4, rouge_l = 0.0, 0.0
            alpha = 1.0
            beta = 1.0
            bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
            rouge_eval = RougeLWithBonus(alpha=alpha, beta=beta, gamma=1.2)

            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                qid = ref['question_id']

                pred_dict[qid] = {}
                for k in ['answers', 'yesno_answers']:
                    if k == 'answers':
                        pred_dict[qid][k] = normalize(pred[k])
                    else:
                        pred_dict[qid][k] = pred[k]

                ref_dict[qid] = {}
                for k in ['source', 'answers', 'yesno_answers', 'entity_answers', 'question_type']:
                    if k == 'answers':
                        ref_dict[qid][k] = normalize(ref[k])
                    else:
                        ref_dict[qid][k] = ref[k]
                for i, e in enumerate(ref_dict[qid]['entity_answers']):
                    ref_dict[qid]['entity_answers'][i] = normalize(e)

            bleu4, rouge_l = calc_metrics(pred_dict,
                    ref_dict,
                    bleu_eval,
                    rouge_eval)
            bleu_rouge = {
                    'ROUGE-L': round(rouge_l* 100, 2),
                    'BLEU-4': round(bleu4 * 100, 2),
            }
            
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge


    def rouge(self, cand, ref):
        string, sub = cand, ref
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        basic_lcs = lengths[len(string)][len(sub)]

        prec = basic_lcs / len(cand) if len(cand) > 0. else 0.
        rec = basic_lcs / len(ref) if len(ref) > 0. else 0.
        gamma = 1.2
        score = 0.0
        if float(rec + gamma**2 * prec) != 0.0:             
            score = ((1 + gamma**2) * prec * rec) / \
                        float(rec + gamma**2 * prec)

        return score


    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len, hack=False):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        if hack:
            span_scores = []
            answers = []
            for p_idx, passage in enumerate(sample['passages']):
                if p_idx >= self.max_p_num:
                    continue
                passage_len = min(self.max_p_len, len(passage['passage_tokens']))
                answer_span, score = self.find_best_answer_for_passage(
                    start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                    end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                    passage_len)

                span_scores.append(score)
                answers.append(''.join(
                    sample['passages'][p_idx]['passage_tokens'][answer_span[0]: answer_span[1] + 1]))

            scores = []
            for out_idx in range(len(answers)):
                vote_score = 0.0
                for in_idx in range(len(answers)):
                    vote_score += 1.0 * self.rouge(answers[out_idx], answers[in_idx]) * math.log(1. + span_scores[in_idx])
                scores.append((span_scores[out_idx] ** (1./4)) * vote_score)
            
            best_answer = ''
            if len(answers) > 0:
                best_answer = answers[scores.index(max(scores))]
        
        else:
            best_p_idx, best_span, best_score = None, None, 0
            for p_idx, passage in enumerate(sample['passages']):
                if p_idx >= self.max_p_num:
                    continue
                passage_len = min(self.max_p_len, len(passage['passage_tokens']))
                answer_span, score = self.find_best_answer_for_passage(
                    start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                    end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                    passage_len)
                if score > best_score:
                    best_score = score
                    best_p_idx = p_idx
                    best_span = answer_span
            if best_p_idx is None or best_span is None:
                best_answer = ''
            else:
                best_answer = ''.join(
                    sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob


    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
        

    
