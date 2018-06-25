import tensorflow as tf

import model_helper
import gnmt_model
from utils import misc_utils as utils

def infer(ckpt, inference_input_file, inference_output_file, hparams):
    """
    Perform translation.
    """
    model_creator = gnmt_model.GNMTModel
    infer_model = model_helper.create_infer_model(model_creator, hparams)

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
        utils.log("Start decoding")
        loaded_infer_model.decode_and_evaluate(
            "infer",
            sess,
            inference_output_file,
            ref_file=None,
            beam_width=hparams.beam_width,
            tgt_eos=hparams.eos,
            num_translations_per_input=hparams.num_translations_per_input)