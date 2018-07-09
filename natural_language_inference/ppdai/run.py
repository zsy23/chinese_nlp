import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import logging
import argparse
import random
import numpy as np
import tensorflow as tf

import misc_utils as utils
import data_utils
from model import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--word_char", type=str, default="word", help="Use word/char to train")

    parser.add_argument("--train_file", type=str, default="", help="Train data file")
    parser.add_argument("--valid_file", type=str, default="", help="Valid data file")
    parser.add_argument("--infer_file", type=str, default="", help="Infer data file")
    parser.add_argument("--question_file", type=str, default="question.csv", help="Question file")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="Output file")
    parser.add_argument("--word_embed_file", type=str, default="word_embed.txt", help="Word embedding file")
    parser.add_argument("--char_embed_file", type=str, default="char_embed.txt", help="Char embedding file")
    parser.add_argument("--model_dir", type=str, default="model", help="Model output directory")
    parser.add_argument("--model_prefix", type=str, default="ppdai", help="Model name prefix")

    parser.add_argument("--hidden_size", type=int, default=150, help="Hidden state size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--init_op", type=str, default="glorot_uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))  
    parser.add_argument("--lr", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")

    parser.add_argument("--epochs", type=int, default=100, help="Epoch number")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed")

    parser.add_argument("--log_file", type=str, default='log', help="Log file")


    return parser.parse_args()


def create_hparams(args):
    hparams = tf.contrib.training.HParams(
        word_char=args.word_char,

        train_file=args.train_file,
        valid_file=args.valid_file,
        infer_file=args.infer_file,
        question_file=args.question_file,
        output_file=args.output_file,
        word_embed_file=args.word_embed_file,
        char_embed_file=args.char_embed_file,
        model_dir=args.model_dir,
        model_prefix=args.model_prefix,

        hidden_size=args.hidden_size,
        dropout=args.dropout,

        batch_size=args.batch_size,
        init_op=args.init_op,
        init_weight=args.init_weight,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,

        epochs=args.epochs,
        random_seed=args.random_seed
    )

    assert hparams.word_char in ['word', 'char'], 'must be word or char'

    best_model_dir = os.path.join(hparams.model_dir, "best")
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    hparams.add_hparam("best_model_dir", best_model_dir)

    if hparams.train_file:
        with open(hparams.train_file, 'r') as fin:
            num_train_steps = len(fin.readlines())
        num_train_steps = int(1.0 * num_train_steps * hparams.epochs / hparams.batch_size)
        hparams.add_hparam("num_train_steps", num_train_steps)

        hparams.add_hparam("best_loss", -1.0)

    return hparams


def train(hparams):
    utils.log('Training ...')

    model_dir = hparams.model_dir

    if hparams.word_char == 'word':
        word_embed, word_vocab = data_utils.get_embed_vocab(hparams.word_embed_file)
        data_utils.init_data('word', word_vocab, hparams.question_file, train_file=hparams.train_file, valid_file=hparams.valid_file)
    else:
        char_embed, char_vocab = data_utils.get_embed_vocab(hparams.char_embed_file)
        data_utils.init_data('char', char_vocab, hparams.question_file, train_file=hparams.train_file, valid_file=hparams.valid_file)
    

    train_graph = tf.Graph()
    utils.log('Creating train model ...')
    with train_graph.as_default(), tf.container("train"):
        train_iterator = data_utils.get_iterator('train', hparams.batch_size, shuffle=True)
        train_model = Model(hparams, 
            word_embed if hparams.word_char == 'word' else char_embed, 
            train_iterator, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    utils.log('Creating eval model ...')
    with eval_graph.as_default(), tf.container("eval"):
        eval_iterator = data_utils.get_iterator('valid', hparams.batch_size, shuffle=False)
        eval_model = Model(hparams,
        word_embed if hparams.word_char == 'word' else char_embed,
        eval_iterator, tf.contrib.learn.ModeKeys.EVAL)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    train_sess = tf.Session(graph=train_graph, config=config_proto)
    eval_sess = tf.Session(graph=eval_graph, config=config_proto)

    log_every_n_batch = 10
    eval_every_n_batch = int(hparams.num_train_steps / hparams.epochs / 3)

    with train_graph.as_default(), tf.container("train"):
        train_sess.run(tf.global_variables_initializer())


    for epoch in range(1, hparams.epochs + 1):
        utils.log('Start epoch {}'.format(epoch))

        total_loss = 0
        total_num = 0
        batch_loss = 0

        train_sess.run(train_iterator.initializer)

        while True:
            try:
                global_step, batch_size, loss, _ = train_model.train(train_sess, hparams.lr)
            except tf.errors.OutOfRangeError:
                utils.log('Finish epoch {}, average train loss is {}'.format(
                    epoch, total_loss / total_num))

                break

            total_loss += loss * batch_size
            total_num += batch_size
            batch_loss += loss

            if global_step > 0 and global_step % log_every_n_batch == 0:
                utils.log('Average loss from batch {} to batch {} is loss {} with lr {}'.format(
                    global_step - log_every_n_batch + 1, global_step, 
                    batch_loss / log_every_n_batch, hparams.lr
                ))
                batch_loss = 0
            
            if global_step > 0 and global_step % eval_every_n_batch == 0:
                train_model.save(train_sess, model_dir, hparams.model_prefix)
                eval(hparams, eval_model, eval_sess, eval_iterator)

    utils.log('Done')


def eval(hparams, model, sess, iterator):
    utils.log('Start eval on dev')

    latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
    assert latest_ckpt, 'No ckpt for eval'
    model.saver.restore(sess, latest_ckpt)
    utils.log('Model parameters restored from {}'.format(latest_ckpt))

    dev_loss = 0
    dev_num = 0

    sess.run(iterator.initializer)

    while True:
        try:
            batch_size, loss = model.eval(sess)
        except tf.errors.OutOfRangeError:
            utils.log('Finish eval on dev, average dev loss is {}'.format(
                dev_loss / dev_num
            ))

            break
        
        dev_loss += loss * batch_size
        dev_num += batch_size

    if hparams.best_loss < 0 or hparams.best_loss > (dev_loss / dev_num):
        hparams.best_loss = (dev_loss / dev_num)
        model.save(sess, hparams.best_model_dir, hparams.model_prefix)

    utils.log('Best loss is {}'.format(hparams.best_loss))


def infer(hparams):
    utils.log('Inferring ...')

    if hparams.word_char == 'word':
        word_embed, word_vocab = data_utils.get_embed_vocab(hparams.word_embed_file)
        data_utils.init_data('word', word_vocab, hparams.question_file, infer_file=hparams.infer_file, test=True)
    else:
        char_embed, char_vocab = data_utils.get_embed_vocab(hparams.char_embed_file)
        data_utils.init_data('char', char_vocab, hparams.question_file, infer_file=hparams.infer_file, test=True)
    

    infer_graph = tf.Graph()
    utils.log('Creating infer model ...')
    with infer_graph.as_default(), tf.container("infer"):
        infer_iterator = data_utils.get_iterator('infer', hparams.batch_size, shuffle=False)
        infer_model = Model(hparams, 
        word_embed if hparams.word_char == 'word' else char_embed,
        infer_iterator, tf.contrib.learn.ModeKeys.EVAL)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    infer_sess = tf.Session(graph=infer_graph, config=config_proto)

    latest_ckpt = tf.train.latest_checkpoint(hparams.best_model_dir)
    assert latest_ckpt, 'No ckpt for eval'
    infer_model.saver.restore(infer_sess, latest_ckpt)
    utils.log('Model parameters restored from {}'.format(latest_ckpt))

    scores = []

    infer_sess.run(infer_iterator.initializer)

    while True:
        try:
            batch_scores = infer_model.infer(infer_sess)
            for score in batch_scores:
                scores.append(score[1])
        except tf.errors.OutOfRangeError:
            with open(hparams.output_file, 'w') as fout:
                fout.write('y_pre\n')
                for score in scores:
                    fout.write(str(score) + '\n')
            
            utils.log('Finish eval on test, result saved to {}'.format(
                hparams.output_file
            ))

            break

    utils.log('Done')


def run(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    logger = logging.getLogger("text_similarity")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.model_dir, args.log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    hparams = create_hparams(args)
    utils.log('Running with hparams : {}'.format(hparams))

    random_seed = hparams.random_seed
    if random_seed is not None and random_seed > 0:
        utils.log('Set random seed to {}'.format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed) 
        tf.set_random_seed(random_seed)

    utils.log('Process {} model'.format(hparams.word_char))

    if hparams.infer_file:
        infer(hparams)
    else:
        train(hparams)


if __name__ == '__main__':
    args = parse_args()
    run(args)
