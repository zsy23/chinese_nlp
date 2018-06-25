import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import logging

import random
import numpy as np

import tensorflow as tf

import train
import infer
from utils import evaluation_utils
from utils import vocab_utils
from utils import misc_utils as utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")  

    # network
    parser.add_argument("--embed_size", type=int, default=1024, help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden state size.")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Network depth.")  
    parser.add_argument("--attention", type=str, default="normed_bahdanau", help="""\
        luong | scaled_luong | bahdanau | normed_bahdanau""")
    parser.add_argument(
        "--output_attention", type="bool", nargs="?", const=True,
        default=True,
        help="""\
        Only used in standard attention_architecture. Whether use attention as
        the cell output at each timestep.
        .\
        """)
    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
        How to warmup learning rates. Options include:
            t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
                exponentiate until the specified lr.\
        """)
    parser.add_argument(
        "--decay_scheme", type=str, default="luong10", help="""\
        How we decay learning rate. Options include:
            luong234: after 2/3 num train steps, we start halving the learning rate
            for 4 times before finishing.
            luong5: after 1/2 num train steps, we start halving the learning rate
            for 5 times before finishing.\
            luong10: after 1/2 num train steps, we start halving the learning rate
            for 10 times before finishing.\
        """)
    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))    
    # data
    parser.add_argument("--src", type=str, default="zh",
                        help="Source suffix, e.g., zh.")
    parser.add_argument("--tgt", type=str, default="en",
                        help="Target suffix, e.g., en.")
    parser.add_argument("--train_prefix", type=str, default="corpus_small",
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default="dev",
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default="test",
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default="model",
                        help="Store model files.")   
    # vocab
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                        default=False,
                        help="""\
        Whether to use the source vocab and embeddings for both source and
        target.\
        """)
    parser.add_argument("--vocab_prefix", type=str, default="vocab_small", help="""\
        Vocab prefix, expect files with src/tgt suffixes.If None, extract from
        train files.\
        """)
    parser.add_argument("--embed_prefix", type=str, default=None, help="""\
        Pretrained embedding prefix, expect files with src/tgt suffixes.
        The embedding files should be Glove formated txt files.\
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
                        help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--source_reverse", type="bool", nargs="?", const=False,
                        default=False, help="Reverse source sequence.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")  
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")   
    # misc
    parser.add_argument("--epochs", type=int, default=5,
                        help="epoch number")
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
        How many training steps to do per external evaluation.  Automatically set
        based on data if None.\
        """)
    parser.add_argument("--random_seed", type=int, default=23,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                        const=True, default=False, help=("""\
                        Average the last N checkpoints for external evaluation.
                        N can be controlled by setting --num_keep_ckpts.\
                        """))
    # inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
        Reference file to compute evaluation scores (if provided).\
        """))
    parser.add_argument("--beam_width", type=int, default=10,
                        help=("""\
        beam width when using beam search decoder. If 0 (default), use standard
        decoder with greedy helper.\
        """))
    parser.add_argument("--length_penalty_weight", type=float, default=1.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--sampling_temperature", type=float,
                        default=0.0,
                        help=("""\
        Softmax sampling temperature for inference decoding, 0.0 means greedy
        decoding. This option is ignored when using beam search.\
        """))
    parser.add_argument("--num_translations_per_input", type=int, default=1,
                        help=("""\
        Number of translations generated for each sentence. This is only used for
        inference.\
        """))

    return parser.parse_args()

def create_hparams(args):
    """
    Create training hparams.
    """

    return tf.contrib.training.HParams(
        # Data
        src=args.src,
        tgt=args.tgt,
        train_prefix=args.train_prefix,
        dev_prefix=args.dev_prefix,
        test_prefix=args.test_prefix,
        vocab_prefix=args.vocab_prefix,
        embed_prefix=args.embed_prefix,
        out_dir=args.out_dir,

        # Networks
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attention=args.attention,
        output_attention=args.output_attention,
        dropout=args.dropout,
        unit_type=args.unit_type,

        # Train
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        init_op=args.init_op,
        init_weight=args.init_weight,
        max_gradient_norm=args.max_gradient_norm,
        learning_rate=args.learning_rate,
        warmup_steps = args.warmup_steps,
        warmup_scheme=args.warmup_scheme,
        decay_scheme=args.decay_scheme,

        # Data constraints
        num_buckets=args.num_buckets,
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        source_reverse=args.source_reverse,

        # Inference
        ckpt=args.ckpt,
        src_max_len_infer=args.src_max_len_infer,
        tgt_max_len_infer=args.tgt_max_len_infer,
        infer_batch_size=args.infer_batch_size,
        beam_width=args.beam_width,
        length_penalty_weight=args.length_penalty_weight,
        sampling_temperature=args.sampling_temperature,
        num_translations_per_input=args.num_translations_per_input,
        inference_input_file=args.inference_input_file,
        inference_output_file=args.inference_output_file,
        inference_ref_file=args.inference_ref_file,

        # Vocab
        sos=args.sos if args.sos else vocab_utils.SOS,
        eos=args.eos if args.eos else vocab_utils.EOS,

        # Misc
        epochs=args.epochs,
        forget_bias=args.forget_bias,
        steps_per_stats=args.steps_per_stats,
        steps_per_external_eval=args.steps_per_external_eval,
        random_seed=args.random_seed,
        share_vocab=args.share_vocab,
        num_keep_ckpts=args.num_keep_ckpts,
        avg_ckpts=args.avg_ckpts
    )

def create_or_load_hparams(
    out_dir, default_hparams):
    """
    Create hparams or load hparams from out_dir.
    """

    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams

        hparams.add_hparam("best_bleu", 0)
        best_bleu_dir = os.path.join(out_dir, "best_bleu")
        hparams.add_hparam("best_bleu_dir", best_bleu_dir)
        os.makedirs(best_bleu_dir)
        hparams.add_hparam("avg_best_bleu", 0)
        best_bleu_dir = os.path.join(hparams.out_dir, "avg_best_bleu")
        hparams.add_hparam("avg_best_bleu_dir", os.path.join(hparams.out_dir, "avg_best_bleu"))
        os.makedirs(best_bleu_dir)

        # Set num_train_steps
        train_src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
        train_tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
        with open(train_src_file, 'r', encoding='utf-8') as f:
            train_src_steps = len(f.readlines())
        with open(train_tgt_file, 'r', encoding='utf-8') as f:
            train_tgt_steps = len(f.readlines())
        hparams.add_hparam("num_train_steps", min([train_src_steps, train_tgt_steps]) * hparams.epochs)

        # Set encoder/decoder layers
        hparams.add_hparam("num_encoder_layers", hparams.num_layers)
        hparams.add_hparam("num_decoder_layers", hparams.num_layers)

        # Set residual layers
        num_encoder_residual_layers = 0
        num_decoder_residual_layers = 0
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1

        # The first unidirectional layer (after the bi-directional layer) in
        # the GNMT encoder can't have residual connection due to the input is
        # the concatenation of fw_cell and bw_cell's outputs.
        num_encoder_residual_layers = hparams.num_encoder_layers - 2

        # Compatible for GNMT models
        if hparams.num_encoder_layers == hparams.num_decoder_layers:
            num_decoder_residual_layers = num_encoder_residual_layers
            
        hparams.add_hparam("num_encoder_residual_layers", num_encoder_residual_layers)
        hparams.add_hparam("num_decoder_residual_layers", num_decoder_residual_layers)

        # Vocab
        # Get vocab file names first
        if hparams.vocab_prefix:
            src_vocab_file = hparams.vocab_prefix + "." + hparams.src
            tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
        else:
            raise ValueError("hparams.vocab_prefix must be provided.")
        # Source vocab
        src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
            src_vocab_file,
            hparams.out_dir,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=vocab_utils.UNK)
        # Target vocab
        if hparams.share_vocab:
            utils.log("Using source vocab for target")
            tgt_vocab_file = src_vocab_file
            tgt_vocab_size = src_vocab_size
        else:
            tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
                tgt_vocab_file,
                hparams.out_dir,
                sos=hparams.sos,
                eos=hparams.eos,
                unk=vocab_utils.UNK)
        hparams.add_hparam("src_vocab_size", src_vocab_size)
        hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
        hparams.add_hparam("src_vocab_file", src_vocab_file)
        hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

        # Pretrained Embeddings:
        hparams.add_hparam("src_embed_file", "")
        hparams.add_hparam("tgt_embed_file", "")
        if hparams.embed_prefix:
            src_embed_file = hparams.embed_prefix + "." + hparams.src
            tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt
            if os.path.exists(src_embed_file):
                hparams.src_embed_file = src_embed_file
            if os.path.exists(tgt_embed_file):
                hparams.tgt_embed_file = tgt_embed_file


    # Save HParams
    utils.save_hparams(out_dir, hparams)

    return hparams

def run(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    logger = logging.getLogger("nmt_zh")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.out_dir, "log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    default_hparams = create_hparams(args)
    # Load hparams.
    hparams = create_or_load_hparams(
        default_hparams.out_dir, default_hparams)

    utils.log('Running with hparams : {}'.format(hparams))

    random_seed = hparams.random_seed
    if random_seed is not None and random_seed > 0:
        utils.log('Set random seed to {}'.format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed) 
        tf.set_random_seed(random_seed)

    if hparams.inference_input_file:
        utils.log('Inferring ...')
        # infer
        trans_file = hparams.inference_output_file
        ckpt = hparams.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(hparams.out_dir)
        utils.log('Use checkpoint: {}'.format(ckpt))
        utils.log('Start infer sentence in {}, output saved to {} ...'.format(
                        hparams.inference_input_file, trans_file))
        infer.infer(ckpt, hparams.inference_input_file, trans_file, hparams)

        # eval
        ref_file = hparams.inference_ref_file
        if ref_file and os.path.exists(trans_file):
            utils.log('Evaluating infer output with reference in {} ...'.format(
                            ref_file))
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                'BLEU')
            utils.log("BLEU: %.1f" % (score, ))
    else:
        utils.log('Training ...')
        train.train(hparams)


if __name__ == '__main__':
    args = parse_args()
    run(args)
