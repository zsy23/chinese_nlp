import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import logging

import random
import numpy as np

import tensorflow as tf

from . import train, infer
from .utils import evaluation_utils
from .utils import vocab_utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")  

    # network
    parser.add_argument("--embed_size", type=int, default=32, help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden state size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")  
    parser.add_argument("--attention", type=str, default="", help="""\
        luong | scaled_luong | bahdanau | normed_bahdanau""")
    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--learning_rate_warmup_steps", type=int, default=0,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--learning_rate_warmup_factor", type=float, default=1.0,
                        help="The inverse decay factor for each warmup step.")
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay")
    parser.add_argument("--decay_factor", type=float, default=0.98,
                        help="How much we decay.")  
    parser.add_argument(
        "--learning_rate_decay_scheme", type=str, default="", help="""\
        If specified, overwrite start_decay_step, decay_steps, decay_factor.
        Options include:
            luong: after 1/2 num train steps, we start halving the learning rate
            for 5 times before finishing.\
        """)
    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))    
    # data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix, e.g., zh.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="Target suffix, e.g., en.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store model files.")  
    parser.add_argument("--summary_dir", type=str, default=None,
                        help="Store summary files.")  
    # vocab
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
        Vocab prefix, expect files with src/tgt suffixes.If None, extract from
        train files.\
        """)
    # sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""\
        Max length of tgt sequences during inference.  Also use to restrict the
        maximum decoding length.\
        """)    
    # default settings works well (rarely need to change)
    parser.add_argument("--unit_type", type=str, default="lstm",
                        help="lstm | gru | layer_norm_lstm")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--source_reverse", type="bool", nargs="?", const=True,
                        default=False, help="Reverse source sequence.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")  
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")   
    # misc
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
        How many training steps to do per external evaluation.  Automatically set
        based on data if None.\
        """)
    parser.add_argument("--scope", type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from FLAGS."))
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                        const=True, default=False,
                        help="Override loaded hparams with values specified")   
    # inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
        Reference file to compute evaluation scores (if provided).\
        """))
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
        beam width when using beam search decoder. If 0 (default), use standard
        decoder with greedy helper.\
        """))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--num_translations_per_input", type=int, default=1,
                        help=("""\
        Number of translations generated for each sentence. This is only used for
        inference.\
        """))

    # log
    parser.add_argument("--log_path", type=str, default=None,
                        help="Log file path.")    

    return parser.parse_args()

def create_hparams(flags):
    """
    Create training hparams.
    """

    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        out_dir=flags.out_dir,

        # Networks
        embed_size=flags.embed_size,
        hidden_size=flags.hidden_size,
        num_layers=flags.num_layers,
        attention=flags.attention,
        dropout=flags.dropout,
        unit_type=flags.unit_type,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        learning_rate_warmup_steps = flags.learning_rate_warmup_steps,
        learning_rate_warmup_factor = flags.learning_rate_warmup_factor,
        start_decay_step=flags.start_decay_step,
        decay_factor=flags.decay_factor,
        decay_steps=flags.decay_steps,
        learning_rate_decay_scheme=flags.learning_rate_decay_scheme,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        source_reverse=flags.source_reverse,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        num_translations_per_input=flags.num_translations_per_input,

        # Vocab
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,

        # Misc
        forget_bias=flags.forget_bias,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        random_seed=flags.random_seed,
    )


def run(args):
    logger = logging.getLogger("nmt_zh")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    hparams = create_hparams(args)

    random_seed = hparams.random_seed
    if random_seed is not None and random_seed > 0:
        logger.info('Set random seed to {}'.format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    out_dir = hparams.out_dir
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    best_bleu_dir = os.path.join(out_dir, "best_bleu")
    hparams.add_hparam("best_bleu_dir", best_bleu_dir)
    if not os.path.exists(best_bleu_dir): 
        os.makedirs(best_bleu_dir)    

    if hparams.inference_input_file:
        logger.info('Inferring ...')
        # infer
        trans_file = hparams.inference_output_file
        ckpt = hparams.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        logger.info('Use checkpoint: {}'.format(ckpt))
        logger.info('Start infer sentence in {}, output saved to {} ...'.format(
                        hparams.inference_input_file, trans_file))
        infer.infer(ckpt, hparams.inference_input_file, trans_file, hparams)

        # eval
        ref_file = hparams.inference_ref_file
        if ref_file and os.path.exists(trans_file):
            logger.info('Evaluating infer output with reference in {} ...'.format(
                            ref_file))
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                'BLEU')
            logger.info("BLEU: %.1f" % (score, ))
    else:
        logger.info('Training ...')
        train.train(hparams)


if __name__ == '__main__':
    args = parse_args()
    run(args)