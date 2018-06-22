import os
import logging
import tensorflow as tf

from .utils import iterator_utils
from .utils import evaluation_utils
from .utils import misc_utils as utils

class GNMTModel(object):
    """
    Sequence-to-sequence dynamic model with GNMT attention architecture.
    """

    def __init__(self,
                hparams,
                mode,
                iterator,
                source_vocab_table,
                target_vocab_table,
                source_vocab_size,
                target_vocab_size,
                reverse_target_vocab_table=None,
                scope=None):

        assert isinstance(iterator, iterator_utils.BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table
        self.src_vocab_size = source_vocab_size
        self.tgt_vocab_size = target_vocab_size
        self.hparams = hparams
        self.logger = logging.getLogger('nmt_zh')

        self.best_bleu = 0

        self.batch_size = tf.size(self.iterator.source_sequence_length)

        # Initializer
        initializer = self.get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        self.build_graph(scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = self.loss
            self.word_count = tf.reduce_sum(
                self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = self.loss
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits = self.logits
            self.sample_words = reverse_target_vocab_table.lookup(
                tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            ## Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(
                self.iterator.target_sequence_length)
        
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        # Gradients and SGD update operation for training the model.
        # Arrage for the embedding vars to appear at the beginning.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            # warm-up
            self.learning_rate = self.get_learning_rate_warmup()
            # decay
            self.learning_rate = self.get_learning_rate_decay()

            # Optimizer
            if hparams.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif hparams.optimizer == "adam":
                assert float(
                    hparams.learning_rate
                ) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
                opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradients
            gradients = tf.gradients(
                self.train_loss,
                params)

            clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                gradients, hparams.max_gradient_norm)
            gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
            gradient_norm_summary.append(
                tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("lr", self.learning_rate),
                tf.summary.scalar("train_loss", self.train_loss),
            ] + gradient_norm_summary)
        
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            if hparams.beam_width > 0:
                self.infer_summary = tf.no_op()
            else:
                attention_images = (self.final_context_state[0].alignment_history.stack())
                # Reshape to (batch, src_seq_len, tgt_seq_len,1)
                attention_images = tf.expand_dims(
                    tf.transpose(attention_images, [1, 2, 0]), -1)
                # Scale to range [0, 255]
                attention_images *= 255
                self.infer_summary = tf.summary.image("attention_images", attention_images)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        self.logger.info("Trainable variables")
        for param in params:
            self.logger.info("%s, %s" % (param.name, str(param.get_shape())))


    def get_initializer(self, init_op, seed=None, init_weight=None):
        """
        Create an initializer. init_weight is only for uniform.
        """

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

    def get_learning_rate_warmup(self):
        """
        Get learning rate warmup.
        """

        hparams = self.hparams
        warmup_steps = hparams.learning_rate_warmup_steps
        warmup_factor = hparams.learning_rate_warmup_factor

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        inv_decay = warmup_factor**(
            tf.to_float(warmup_steps - self.global_step))

        return tf.cond(
            self.global_step < hparams.learning_rate_warmup_steps,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def get_learning_rate_decay(self):
        """
        Get learning rate decay.
        """

        hparams = self.hparams
        if (hparams.learning_rate_decay_scheme and
            hparams.learning_rate_decay_scheme == "luong"):
            start_decay_step = int(hparams.num_train_steps / 2)
            decay_steps = int(hparams.num_train_steps / 10)  # decay 5 times
            decay_factor = 0.5
        else:
            start_decay_step = hparams.start_decay_step
            decay_steps = hparams.decay_steps
            decay_factor = hparams.decay_factor

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def build_graph(self, scope):
        
        self.logger.info('Creating {} graph ...'.format(self.mode))

        dtype = tf.float32

        with tf.variable_scope(scope or "gnmt", dtype=dtype):
            self.build_encoder()
            self.build_decoder()

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                self.compute_loss()
            else:
                self.loss = None


    def compute_loss(self):
        """
        Compute optimization loss.
        """

        target_output = self.iterator.target_output
        max_time = target_output.shape[1].value or tf.shape(target_output)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=self.logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=self.logits.dtype)

        self.loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)


    def make_cell(self, unit_type, num_units, forget_bias, dropout,
                    mode, residual_connection=False):

        """
        Create an instance of a single RNN cell.
        """

        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        # Cell Type
        if unit_type == "lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=forget_bias)
        elif unit_type == "gru":
            cell = tf.contrib.rnn.GRUCell(num_units)
        elif unit_type == "layer_norm_lstm":
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units,
                forget_bias=forget_bias,
                layer_norm=True)
        else:
            raise ValueError("Unknown unit type %s!" % unit_type)

        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell, input_keep_prob=(1.0 - dropout))

        # Residual
        if residual_connection:
            cell = tf.contrib.rnn.ResidualWrapper(cell)

        return cell


    def make_cell_list(self, unit_type, num_units, num_layers, num_residual_layers,
                forget_bias, dropout, mode):
        """
        Create a list of RNN cells.
        """

        cell_list = []
        for i in range(num_layers):
            cell = self.make_cell(
                unit_type=unit_type,
                num_units=num_units,
                forget_bias=forget_bias,
                dropout=dropout,
                mode=mode,
                residual_connection=(i >= num_layers - num_residual_layers)
            )
            cell_list.append(cell)

        return cell_list

    def create_rnn_cell(self, unit_type, num_units, num_layers, num_residual_layers,
                        forget_bias, dropout, mode):

        cell_list = self.make_cell_list(
            unit_type=unit_type,
            num_units=num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
        )

        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)
        

    def build_encoder(self):

        iterator = self.iterator
        hparams = self.hparams
        num_bi_layers = 1
        num_bi_residual_layers = 0
        num_uni_layers = hparams.num_layers - num_bi_layers
        if hparams.residual and hparams.num_layers > 1:
            num_uni_residual_layers = hparams.num_layers - 2
        else:
            num_uni_residual_layers = 0

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype

            with tf.variable_scope("embedding") as scope:
                self.embedding_encoder = tf.get_variable(
                    "embedding_encoder", 
                    [self.src_vocab_size, hparams.embed_size], dtype)

                self.encoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_encoder,
                    iterator.source)
            
            with tf.variable_scope("bidirectional_rnn") as scope:
                fw_cell = self.create_rnn_cell(
                    unit_type=hparams.unit_type,
                    num_units=hparams.hidden_size,
                    num_layers=num_bi_layers,
                    num_residual_layers=num_bi_residual_layers,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode)
                bw_cell = self.create_rnn_cell(
                    unit_type=hparams.unit_type,
                    num_units=hparams.hidden_size,
                    num_layers=num_bi_layers,
                    num_residual_layers=num_bi_residual_layers,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode)

                self.bi_outputs, self.bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    self.encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=iterator.source_sequence_length,
                    swap_memory=True)

                self.bi_outputs = tf.concat(self.bi_outputs, -1)

            with tf.variable_scope("stack_rnn") as scope:
                uni_cell = self.create_rnn_cell(
                    unit_type=hparams.unit_type,
                    num_units=hparams.hidden_size,
                    num_layers=num_uni_layers,
                    num_residual_layers=num_uni_residual_layers,
                    forget_bias=hparams.forget_bias,
                    dropout=hparams.dropout,
                    mode=self.mode)
                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                    uni_cell,
                    self.bi_outputs,
                    dtype=dtype,
                    sequence_length=iterator.source_sequence_length,
                    swap_memory=True)
                
            
            # Pass all encoder state except the first bi-directional layer's state to decoder.
            self.encoder_state = (self.bi_state[1],) + (
                (self.encoder_state,) if num_uni_layers == 1 else self.encoder_state)
    

    def create_attention_mechanism(self, attention_option, num_units, memory,
                                source_sequence_length):
        """
        Create attention mechanism based on the attention_option.
        """

        if attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length)
        elif attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                scale=True)
        elif attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units, 
                memory, 
                memory_sequence_length=source_sequence_length)
        elif attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                normalize=True)
        else:
            raise ValueError("Unknown attention option %s" % attention_option)

        return attention_mechanism


    def make_decoder_cell(self):
        hparams = self.hparams

        memory = self.encoder_outputs
        source_sequence_length = self.iterator.source_sequence_length
        beam_width = hparams.beam_width
        if hparams.residual and hparams.num_layers > 1:
            num_residual_layers = hparams.num_layers - 2
        else:
            num_residual_layers = 0

        if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                self.encoder_state, multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = self.create_attention_mechanism(
            hparams.attention, hparams.hidden_size, memory, source_sequence_length)
        
        cell_list = self.make_cell_list(
            unit_type=hparams.unit_type,
            num_units=hparams.hidden_size,
            num_layers=hparams.num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            mode=self.mode)
        
        # Only wrap the bottom layer with the attention mechanism.
        attention_cell = cell_list.pop(0)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                            beam_width == 0)

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            attention_cell,
            attention_mechanism,
            attention_layer_size=None,  # don't use attenton layer.
            output_attention=False,
            alignment_history=alignment_history,
            name="attention")
        
        cell = GNMTAttentionMultiCell(attention_cell, cell_list)

        decoder_initial_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(
                cell.zero_state(batch_size, tf.float32), encoder_state))

        return cell, decoder_initial_state


    def build_decoder(self):

        iterator = self.iterator
        hparams = self.hparams

        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), 
                            tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                            tf.int32)
        
        # maximum_iteration: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        
        with tf.variable_scope("decoder") as scope:
            dtype = scope.dtype

            with tf.variable_scope("embedding") as scope:
                self.embedding_decoder = tf.get_variable(
                    "embedding_decoder", 
                    [self.tgt_vocab_size, hparams.embed_size], dtype)

            cell, decoder_initial_state = self.make_decoder_cell()

            with tf.variable_scope("output_projection"):
                self.output_layer = tf.layers.Dense(
                    self.tgt_vocab_size, 
                    use_bias=False, 
                    name="output_projection")

            ## train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                self.decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, 
                    iterator.target_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_emb_inp, 
                    iterator.target_sequence_length)

                # Decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state)

                # Dynamic decoding
                outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    swap_memory=True)

                self.sample_id = outputs.sample_id

                # Note: there's a subtle difference here between train and inference.
                # We could have set output_layer when create my_decoder
                #   and shared more code between train and inference.
                # We chose to apply the output_layer to all timesteps for speed:
                #   10% improvements for small models & 20% for larger ones.
                # If memory is a concern, we should apply output_layer per timestep.
                self.logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    # Helper
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder, start_tokens, end_token)

                    # Decoder
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer
                    )

                # Dynamic decoding
                outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=maximum_iterations,
                    swap_memory=True)

                if beam_width > 0:
                    self.logits = tf.no_op()
                    self.sample_id = outputs.predicted_ids
                else:
                    self.logits = outputs.rnn_output
                    self.sample_id = outputs.sample_id

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                        self.train_loss,
                        self.predict_count,
                        self.train_summary,
                        self.global_step,
                        self.word_count,
                        self.batch_size])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                        self.predict_count,
                        self.batch_size])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.infer_logits, 
                        self.infer_summary, 
                        self.sample_id, 
                        self.sample_words
        ])

    def compute_perplexity(self, sess, name):
        """
        Compute perplexity of the output of the model.
        """

        total_loss = 0
        total_predict_count = 0

        while True:
            try:
                loss, predict_count, batch_size = self.eval(sess)
                total_loss += loss * batch_size
                total_predict_count += predict_count
            except tf.errors.OutOfRangeError:
                break

        perplexity = utils.safe_exp(total_loss / total_predict_count)
        self.logger.info("{} perplexity: %.2f".format(name, perplexity))
        
        return perplexity

    def decode_and_evaluate(self,
                            name,
                            sess,
                            trans_file,
                            ref_file,
                            beam_width,
                            tgt_eos,
                            num_translations_per_input=1,
                            decode=True):
        """
        Decode a test set and compute a score according to the evaluation task.
        """

        # Decode
        if decode:
            self.logger.info("Decoding to output {}.".format(trans_file))

            num_sentences = 0
            with open(trans_file, 'w', encoding='utf-8') as trans_f:
                trans_f.write("")  # Write empty string to ensure file is created.

                num_translations_per_input = max(
                    min(num_translations_per_input, beam_width), 1)
                while True:
                    try:
                        _, _, _, nmt_outputs = self.infer(sess)
                        if beam_width == 0:
                            nmt_outputs = np.expand_dims(nmt_outputs, 0)

                        batch_size = nmt_outputs.shape[1]
                        num_sentences += batch_size

                        for sent_id in range(batch_size):
                            for beam_id in range(num_translations_per_input):
                                translation = utils.get_translation(
                                    nmt_outputs[beam_id],
                                    sent_id,
                                    tgt_eos=tgt_eos)
                                trans_f.write(translation + "\n")
                    except tf.errors.OutOfRangeError:
                        self.logger.info(
                            "Done, num sentences %d, num translations per input %d" %
                            (num_sentences, num_translations_per_input))
                        break

        # Evaluation
        evaluation_scores = {}
        if ref_file and os.path.exists(trans_file):
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                'BLEU')
            evaluation_scores['BLEU'] = score
            self.logger.info("%s BLEU: %.1f" % (name, score))

        return evaluation_scores


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """
    A MultiCell with GNMT attention style.
    """

    def __init__(self, attention_cell, cells):
        """
        Creates a GNMTAttentionMultiCell.
        """
        cells = [attention_cell] + cells
        super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """
        Run the cell with bottom layer's attention copied to all upper layers.
        """

        if not tf.contrib.framework.nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "gnmt_attention_multi_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):
                    cell = self._cells[i]
                    cur_state = state[i]

                    if not isinstance(cur_state, tf.contrib.rnn.LSTMStateTuple):
                        raise TypeError("`state[{}]` must be a LSTMStateTuple".format(i))

                    cur_state = cur_state._replace(h=tf.concat(
                        [cur_state.h, new_attention_state.attention], 1))

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)
        