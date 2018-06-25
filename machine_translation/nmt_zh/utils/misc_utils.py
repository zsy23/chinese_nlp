import os
import json
import math
import logging
import tensorflow as tf

logger = logging.getLogger('nmt_zh')

def log(msg):
    global logger

    logger.info(msg)

def safe_exp(value):
    """
    Exponentiation with catching of overflow error.
    """

    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans

def load_hparams(model_dir):
    """
    Load hparams from an existing model directory.
    """

    hparams_file = os.path.join(model_dir, "hparams")
    if os.path.exists(hparams_file):
        log("Loading hparams from {}".format(hparams_file))
        with open(hparams_file, "r", encoding='utf-8') as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                log("Can't load hparams file")
                return None
        return hparams
    else:
        return None

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
    log("Saving hparams to {}".format(hparams_file))
    with open(hparams_file, "w") as f:
        f.write(hparams.to_json())

def format_results(name, ppl, scores):
    """
    Format results.
    """

    result_str = ""
    if ppl:
        result_str = "%s ppl %.2f" % (name, ppl)
    if scores:
        if result_str:
            result_str += ", %s BLEU %.1f" % (name, scores['BLEU'])
        else:
            result_str = "%s BLEU %.1f" % (name, scores['BLEU'])
    return result_str

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
    
    translation = (b" ".join(output)).decode('utf-8')

    return translation