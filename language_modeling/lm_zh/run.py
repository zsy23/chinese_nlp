import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import tensorflow as tf

from params import Params
from data_utils import Dataset
from language_model import LM

flags = tf.flags
flags.DEFINE_string("model_dir", "ptb", "Model directory.")
'''
log_path = $(model_dir)/log
conf_path = $(model_dir)/conf
summary_dir = $(model_dir)/summary
'''
flags.DEFINE_string("data_prefix", "data/ptb/ptb", 
                    "Data path prefix(need $(data_prefix).train.txt, $(data_prefix).valid.txt, $(data_prefix).test.txt).")
flags.DEFINE_string("mode", "train", "Whether to 'train' or 'test' model.")
FLAGS = flags.FLAGS

model_dir = FLAGS.model_dir
log_path = os.path.join(model_dir, 'log')
conf_path = os.path.join(model_dir, 'conf')
summary_dir = os.path.join(model_dir, 'summary')

default_config = Params(
    batch_size=32,
    epoch=13,
    init_scale=1.0,
    embed_size=200,
    num_sampled=0,
    lr=1.0,
    lr_decay=0.5,
    lr_keep_epoch=4,
    keep_prob=1.0,
    num_steps=20,
    num_layers=2,
    max_grad_norm=5.0,

    algo='gcnn',

    # lstm
    hidden_size=200,

    # gcnn
    filter_w=5,
    filter_size=64,
    block_size=1
)

def test(config, dataset, model_dir, summary_dir):
    logger = logging.getLogger('lm_zh')
    config.keep_prob = 1.0
    config.num_sampled = 0
    logger.info('Build graph ...')
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope('model', initializer=initializer):
        model = LM(dataset, config, model_dir, summary_dir)
        logger.info('Restore model ...')
        model.restore()
        logger.info('Start test model ...')
        model.test()
    logger.info('Test done')

def train(config, dataset, model_dir, summary_dir):
    logger = logging.getLogger('lm_zh')
    logger.info('Build graph ...')
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope('model', initializer=initializer):
        model = LM(dataset, config, model_dir, summary_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('Restore model from checkpoint ...')
            model.saver.restore(model.sess, ckpt.model_checkpoint_path)
        logger.info('Start train model ...')
        model.train()
    logger.info('Train done')

def main():
    assert os.path.exists(model_dir)
    assert os.path.exists(conf_path)
    assert os.path.exists(summary_dir)    
    assert os.path.exists(FLAGS.data_prefix + '.train.txt') and \
            os.path.exists(FLAGS.data_prefix + '.valid.txt') and \
            os.path.exists(FLAGS.data_prefix + '.test.txt')
    assert FLAGS.mode in ['train', 'test']

    logger = logging.getLogger("lm_zh")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    logger.info('Parse config file ...')
    config = default_config.parse(conf_path)
    logger.info('Running with config: {}'.format(config.items))

    if FLAGS.mode == 'test':
        config.batch_size *= 2

    logger.info('Build vocab and dataset ...')
    dataset = Dataset(FLAGS.data_prefix, config.num_steps, config.batch_size, train=(FLAGS.mode=='train'))

    print('Use algo:', config.algo)

    if FLAGS.mode == 'train':
        train(config, dataset, model_dir, summary_dir)
    elif FLAGS.mode == 'test':
        test(config, dataset, model_dir, summary_dir)


if __name__ == '__main__':
    main()