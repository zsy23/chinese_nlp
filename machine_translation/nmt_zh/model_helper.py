import logging
import collections
import tensorflow as tf

from .utils import iterator_utils
from .utils import vocab_utils


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(
    model_creator, hparams, scope=None):

    """Create train graph, model, and iterator."""

    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.src)
    tgt_vocab_file = "%s.%s" % (hparams.vocabprefix, hparams.tgt)

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table, src_vocab_size, tgt_vocab_size = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
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
            target_vocab_table=tgt_vocab_table,
            source_vocab_size=src_vocab_size,
            target_vocab_size=tgt_vocab_size,
            scope=scope)

    return TrainModel(
                graph=graph,
                model=model,
                iterator=iterator)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
    pass


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):

    """Create train graph, model, src/tgt file holders, and iterator."""

    src_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.src)
    tgt_vocab_file = "%s.%s" % (hparams.vocabprefix, hparams.tgt)

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "eval"):
        src_vocab_table, tgt_vocab_table, src_vocab_size, tgt_vocab_size = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file)
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
            target_vocab_table=tgt_vocab_table,
            source_vocab_size=src_vocab_size,
            target_vocab_size=tgt_vocab_size,
            scope=scope)
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


def create_infer_model(model_creator, hparams, scope=None, extra_args=None):

    """Create inference model."""

    src_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.src)
    tgt_vocab_file = "%s.%s" % (hparams.vocabprefix, hparams.tgt)

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table, src_vocab_size, tgt_vocab_size = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file)
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
            source_vocab_size=src_vocab_size,
            target_vocab_size=tgt_vocab_size,
            reverse_target_vocab_table=reverse_tgt_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)

def load_model(model, ckpt, session, name):
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    logger = logging.getLogger('nmt_zh')
    logger.info("Load {} model parameters from {}".format(name, ckpt))
    return model

def create_or_load_model(model, model_dir, session, name):

    """Create translation model and initialize or load parameters in session."""

    logger = logging.getLogger('nmt_zh')

    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.saver.restore(session, latest_ckpt)
        session.run(tf.tables_initializer())
        logger.info("Load {} model parameters from {}".format(name, latest_ckpt))
    else:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        logger.info("Create {} model with fresh parameters".format(name))

    global_step = model.global_step.eval(session=session)

    return model, global_step