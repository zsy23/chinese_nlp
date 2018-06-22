import os
import math
import time
import random
import logging
import tensorflow as tf

from . import model_helper
from . import gnmt_model
from .utils import misc_utils as utils

def train(hparams, scope=None):
    logger = logging.getLogger('nmt_zh')

    model_creator = gnmt_model.GNMTModel
    train_model = model_helper.create_train_model(model_creator, hparams, scope)
    eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    train_sess = tf.Session(graph=train_model.graph, config=config_proto)
    eval_sess = tf.Session(graph=eval_model.graph, config=config_proto)
    infer_sess = tf.Session(graph=infer_model.graph, config=config_proto)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, hparams.out_dir, train_sess, "train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(hparams.summary_dir, train_model.graph)

    # Preload data for sample decoding.
    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    sample_src_data = utils.load_data(dev_src_file)
    sample_tgt_data = utils.load_data(dev_tgt_file)

    # First evaluation
    result_summary, _, _, _, _, _ = run_full_eval(
        hparams.out_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data)
    logger.info('First evaluation: {}'.format(result_summary))

    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    # This is the training loop.
    step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
    checkpoint_total_count = 0.0
    speed, train_ppl = 0.0, 0.0

    logger.info(
        "Start step %d, lr %g" %
        (global_step, loaded_train_model.learning_rate.eval(session=train_sess)))

    # Initialize all of the iterators
    train_sess.run(train_model.iterator.initializer)

    model_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_external_eval = hparams.steps_per_external_eval
    steps_per_eval = 10 * steps_per_stats
    if not steps_per_external_eval:
        steps_per_external_eval = 5 * steps_per_eval

    while global_step < num_train_steps:
        ### Run a step ###
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            (_, step_loss, step_predict_count, step_summary, global_step,
            step_word_count, batch_size) = step_result
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            logger.info(
                "# Finished an epoch, step %d. Perform external evaluation" %
                (global_step, ))
            run_sample_decode(infer_model, infer_sess,
                                model_dir, hparams, summary_writer, sample_src_data,
                                sample_tgt_data)
            dev_scores, test_scores, _ = run_external_eval(
                infer_model, infer_sess, model_dir,
                hparams, summary_writer)
            train_sess.run(
                train_model.iterator.initializer)
            continue

        # Write step summary.
        summary_writer.add_summary(step_summary, global_step)

        # update statistics
        step_time += (time.time() - start_time)

        checkpoint_loss += (step_loss * batch_size)
        checkpoint_predict_count += step_predict_count
        checkpoint_total_count += float(step_word_count)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step

            # Print statistics for the previous epoch.
            avg_step_time = step_time / steps_per_stats
            train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
            speed = checkpoint_total_count / (1000 * step_time)
            logger.info(
                "global step %d lr %g "
                "step-time %.2fs wps %.2fK ppl %.2f best BLEU %.2f" %
                (global_step,
                loaded_train_model.learning_rate.eval(session=train_sess),
                avg_step_time, speed, train_ppl, loaded_train_model.best_bleu))
            if math.isnan(train_ppl):
                break

            # Reset timer and loss.
            step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
            checkpoint_total_count = 0.0

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step

            logger.info("Save eval, global step %d" % global_step)
            utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(model_dir, "translate.ckpt"),
                global_step=global_step)

            # Evaluate on dev/test
            run_sample_decode(infer_model, infer_sess,
                                model_dir, hparams, summary_writer, sample_src_data,
                                sample_tgt_data)
            dev_ppl, test_ppl = run_internal_eval(
                eval_model, eval_sess, model_dir, hparams, summary_writer)

        if global_step - last_external_eval_step >= steps_per_external_eval:
            last_external_eval_step = global_step

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(model_dir, "translate.ckpt"),
                global_step=global_step)
            run_sample_decode(infer_model, infer_sess,
                                model_dir, hparams, summary_writer, sample_src_data,
                                sample_tgt_data)
            dev_scores, test_scores, _ = run_external_eval(
                infer_model, infer_sess, model_dir,
                hparams, summary_writer)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(model_dir, "translate.ckpt"),
        global_step=global_step)

    result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = run_full_eval(
        model_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data)
    logger.info(
        "Final, step %d lr %g "
        "step-time %.2f wps %.2fK ppl %.2f, %s" %
        (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
        avg_step_time, speed, train_ppl, result_summary))
    logger.info("Done training!")

    logger.inf("Start evaluating saved best models.")
    best_model_dir = hparams.best_bleu_dir
    result_summary, best_global_step, _, _, _, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    logger.info("Best BLEU, step %d "
                    "step-time %.2f wps %.2fK, %s" %
                    (best_global_step, avg_step_time, speed,
                    result_summary))

    summary_writer.close()
    return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)


def format_results(name, ppl, scores):
    """
    Format results.
    """

    result_str = "%s ppl %.2f" % (name, ppl)
    if scores:
        result_str += ", %s BLEU %.1f" % (name, scores['BLEU'])

    return result_str

def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data):
    """
    Wrapper for running sample_decode, internal_eval and external_eval.
    """

    run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                        sample_src_data, sample_tgt_data)
    dev_ppl, test_ppl = run_internal_eval(
        eval_model, eval_sess, model_dir, hparams, summary_writer)
    dev_scores, test_scores, global_step = run_external_eval(
        infer_model, infer_sess, model_dir, hparams, summary_writer)

    result_summary = format_results("dev", dev_ppl, dev_scores)
    if hparams.test_prefix:
        result_summary += ", " + format_results("test", test_ppl, test_scores)

    return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl

def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
    """
    Sample decode a random sentence from src_data.
    """

    logger = logging.getLogger('nmt_zh')

    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    # Pick a sentence and decode."""
    decode_id = random.randint(0, len(src_data) - 1)

    iterator_feed_dict = {
        infer_model.src_placeholder: [src_data[decode_id]],
        infer_model.batch_size_placeholder: 1,
    }
    infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)

    _, attention_summary, _, nmt_outputs = loaded_infer_model.infer(infer_sess)

    if hparams.beam_width > 0:
        # get the top translation.
        nmt_outputs = nmt_outputs[0]

    translation = utils.get_translation(
        nmt_outputs,
        sent_id=0,
        tgt_eos=hparams.eos)

    logger.info("Sample src: {}".format(src_data[decode_id]))
    logger.info("Sample ref: {}".format(tgt_data[decode_id]))
    logger.info("NMT output: {}".format(translation))

    # Summary
    if attention_summary is not None:
        summary_writer.add_summary(attention_summary, global_step)

def internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
    """
    Computing perplexity.
    """

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    ppl = model.compute_perplexity(sess, label)
    utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)

    return ppl

def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer):
    """
    Compute internal evaluation (perplexity) for both dev / test.
    """

    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_model.model, model_dir, eval_sess, "eval")

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: dev_src_file,
        eval_model.tgt_file_placeholder: dev_tgt_file
    }

    dev_ppl = internal_eval(loaded_eval_model, global_step, eval_sess,
                            eval_model.iterator, dev_eval_iterator_feed_dict,
                            summary_writer, "dev")
    test_ppl = None
    if hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tgt_file
        }
        test_ppl = internal_eval(loaded_eval_model, global_step, eval_sess,
                                eval_model.iterator, test_eval_iterator_feed_dict,
                                summary_writer, "test")
    return dev_ppl, test_ppl

def external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best):
    """
    External evaluation such as BLEU and ROUGE scores.
    """

    logger = logging.getLogger('nmt_zh')

    out_dir = hparams.out_dir
    decode = global_step > 0
    if decode:
        logger.info("External evaluation, global step %d" % global_step)

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = model.decode_and_evaluate(
        label,
        sess,
        output,
        ref_file=tgt_file,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos,
        decode=decode)
    # Save on best metrics
    if decode:
        utils.add_summary(summary_writer, global_step, "%s_BLEU" % (label, ),
                            scores['BLEU'])
        # metric: larger is better
        if save_on_best and scores['BLEU'] > model.best_bleu:
            model.best_bleu = scores['BLEU']
            model.saver.save(
                sess,
                os.path.join(
                    hparams.best_bleu_dir, "translate.ckpt"),
                global_step=model.global_step)
        utils.save_hparams(out_dir, hparams)
    return scores

def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True):
    """
    Compute external evaluation (bleu, rouge, etc.) for both dev / test.
    """

    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: utils.load_data(dev_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    dev_scores = external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        dev_infer_iterator_feed_dict,
        dev_tgt_file,
        "dev",
        summary_writer,
        save_on_best=save_best_dev)

    test_scores = None
    if hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: utils.load_data(test_src_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size,
        }
        test_scores = external_eval(
            loaded_infer_model,
            global_step,
            infer_sess,
            hparams,
            infer_model.iterator,
            test_infer_iterator_feed_dict,
            test_tgt_file,
            "test",
            summary_writer,
            save_on_best=False)

    return dev_scores, test_scores, global_step