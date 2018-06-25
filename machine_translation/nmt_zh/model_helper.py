import six, os
import collections
import tensorflow as tf
import numpy as np

from utils import iterator_utils
from utils import vocab_utils
from utils import misc_utils as utils


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(
    model_creator, hparams):

    """Create train graph, model, and iterator."""

    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container("train"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len)

        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table)

    return TrainModel(
                graph=graph,
                model=model,
                iterator=iterator)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
    pass


def create_eval_model(model_creator, hparams):

    """Create train graph, model, src/tgt file holders, and iterator."""

    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container("eval"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table)
    return EvalModel(
                graph=graph,
                model=model,
                src_file_placeholder=src_file_placeholder,
                tgt_file_placeholder=tgt_file_placeholder,
                iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, hparams):

    """Create inference model."""

    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container("infer"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        reverse_tgt_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocab_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(
            src_placeholder)
        iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            src_max_len=hparams.src_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_table)
    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)

def load_model(model, ckpt, session, name):
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    utils.log("Load {} model parameters from {}".format(name, ckpt))
    return model

def create_or_load_model(model, model_dir, session, name):

    """Create translation model and initialize or load parameters in session."""

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.saver.restore(session, latest_ckpt)
        session.run(tf.tables_initializer())
        utils.log("Load {} model parameters from {}".format(name, latest_ckpt))
    else:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.log("Create {} model with fresh parameters".format(name))

    global_step = model.global_step.eval(session=session)

    return model, global_step

def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
    """
    Average the last N checkpoints in the model_dir.
    """

    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if not checkpoint_state:
        utils.log("No checkpoint file found in directory: {}".format(model_dir))
        return None

    # Checkpoints are ordered from oldest to newest.
    checkpoints = (
        checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

    if len(checkpoints) < num_last_checkpoints:
        utils.log(
            "Skipping averaging checkpoints because not enough checkpoints is "
            "avaliable."
        )
        return None

    avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
    if not os.path.exists(avg_model_dir):
        utils.log(
            "Creating new directory {} for saving averaged checkpoints." .format(
            avg_model_dir))
        os.makedirs(avg_model_dir)

    utils.log("Reading and averaging variables in checkpoints:")
    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if name != global_step_name:
            var_values[name] = np.zeros(shape)

    for checkpoint in checkpoints:
        utils.log("{}".format(checkpoint))
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor

    for name in var_values:
        var_values[name] /= len(checkpoints)

    # Build a graph with same variables in the checkpoints, and save the averaged
    # variables into the avg_model_dir.
    with tf.Graph().as_default():
        tf_vars = [
            tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
            for v in var_values
        ]

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        global_step_var = tf.Variable(
            global_step, name=global_step_name, trainable=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                                    six.iteritems(var_values)):
                sess.run(assign_op, {p: value})

        # Use the built saver to save the averaged checkpoint. Only keep 1
        # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
        saver.save(
            sess,
            os.path.join(avg_model_dir, "translate.ckpt"))

    return avg_model_dir