import logging
import tensorflow as tf

from . import model_helper
from . import gnmt_model
from .utils import misc_utils as utils

def infer(ckpt, inference_input_file, trans_file, hparams, scope=None):
    logger = logging.getLogger('nmt_zh')

    model_creator = gnmt_model.GNMTModel
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

    # Read data
    infer_data = utils.load_data(inference_input_file)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(
        graph=infer_model.graph, config=config_proto) as sess:

        loaded_infer_model = model_helper.load_model(
            infer_model.model, ckpt, sess, "infer")
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })
        # Decode
        logger.info("Start decoding")
        loaded_infer_model.decode_and_evaluate(
            "infer",
            sess,
            trans_file,
            ref_file=None,
            beam_width=hparams.beam_width,
            tgt_eos=hparams.eos,
            num_translations_per_input=hparams.num_translations_per_input)