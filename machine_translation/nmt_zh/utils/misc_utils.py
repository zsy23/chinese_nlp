import os
import math
import logging
import tensorflow as tf

def safe_exp(value):
    """
    Exponentiation with catching of overflow error.
    """

    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans

def load_data(inference_input_file):
    """
    Load inference data.
    """

    with open(inference_input_file, 'r', encoding='utf-8') as f:
        inference_data = f.read().splitlines()

    return inference_data

def add_summary(summary_writer, global_step, tag, value):
    """
    Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)

def save_hparams(out_dir, hparams):
    """
    Save hparams.
    """

    hparams_file = os.path.join(out_dir, "hparams")
    logger = logging.getLogger('nmt_zh')
    logger.info("Saving hparams to {}".format(hparams_file))
    with open(hparams_file, "w") as f:
        f.write(hparams.to_json())

def get_translation(nmt_outputs, sent_id, tgt_eos):
    """
    Given decoding output, turn to text.
    """
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")

    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]
    
    translation = " ".join(output)

    return translation