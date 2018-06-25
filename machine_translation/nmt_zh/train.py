import os
import math
import time
import random
import tensorflow as tf

import model_helper
import gnmt_model
from utils import misc_utils as utils

def init_stats():
    """
    Initialize statistics that we want to accumulate.
    """

    return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
            "total_count": 0.0, "grad_norm": 0.0}

def update_stats(stats, start_time, step_result):
    """
    Update stats: write summary and accumulate statistics.
    """

    (_, step_loss, step_predict_count, step_summary, global_step,
    step_word_count, batch_size, grad_norm, learning_rate) = step_result

    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)
    stats["grad_norm"] += grad_norm

    return global_step, learning_rate, step_summary

def process_stats(stats, info, global_step, steps_per_stats):
    """
    Update info and check for overflow.
    """

    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["train_ppl"] = utils.safe_exp(stats["loss"] / stats["predict_count"])
    info["speed"] = stats["total_count"] / (1000 * stats["step_time"])

    # Check for overflow
    is_overflow = False
    train_ppl = info["train_ppl"]
    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
        utils.log("step %d overflow, stop early" % (global_step,))
        is_overflow = True

    return is_overflow

def print_step_info(prefix, global_step, info, result_summary):
    """
    Print all info at the current global step.
    """

    utils.log(
        "%sstep %d lr %g step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s" %
        (prefix, global_step, info["learning_rate"], info["avg_step_time"],
        info["speed"], info["train_ppl"], info["avg_grad_norm"], result_summary))

def train(hparams, scope=None):

    model_dir = hparams.out_dir
    avg_ckpts = hparams.avg_ckpts
    steps_per_stats = hparams.steps_per_stats
    steps_per_external_eval = hparams.steps_per_external_eval
    steps_per_eval = 10 * steps_per_stats
    if not steps_per_external_eval:
        steps_per_external_eval = 5 * steps_per_eval
    summary_name = "summary"

    model_creator = gnmt_model.GNMTModel
    train_model = model_helper.create_train_model(model_creator, hparams)
    eval_model = model_helper.create_eval_model(model_creator, hparams)
    infer_model = model_helper.create_infer_model(model_creator, hparams)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    train_sess = tf.Session(graph=train_model.graph, config=config_proto)
    eval_sess = tf.Session(graph=eval_model.graph, config=config_proto)
    infer_sess = tf.Session(graph=infer_model.graph, config=config_proto)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, model_dir, train_sess, "train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(model_dir, summary_name), train_model.graph)

    # Preload data for sample decoding.
    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    sample_src_data = utils.load_data(dev_src_file)
    sample_tgt_data = utils.load_data(dev_tgt_file)

    # First evaluation
    result_summary, _, _ = run_full_eval(
        model_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data, avg_ckpts)
    utils.log('First evaluation: {}'.format(result_summary))

    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    # This is the training loop.
    stats = init_stats()
    info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
            "avg_grad_norm": 0.0,
            "learning_rate": loaded_train_model.learning_rate.eval(
                session=train_sess)}
    utils.log("Start step %d, lr %g" %
                    (global_step, info["learning_rate"]))

    # Initialize all of the iterators
    train_sess.run(train_model.iterator.initializer)

    epoch = 1

    while True:
        ### Run a step ###
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            utils.log(
                "Finished epoch %d, step %d. Perform external evaluation" %
                (epoch, global_step))
            run_sample_decode(infer_model, infer_sess,
                                model_dir, hparams, summary_writer, sample_src_data,
                                sample_tgt_data)
            run_external_eval(
                infer_model, infer_sess, model_dir,
                hparams, summary_writer)
            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                                    summary_writer, global_step)
            train_sess.run(
                train_model.iterator.initializer)

            if epoch < hparams.epochs:
                epoch += 1
                continue
            else:
                break

        # Process step_result, accumulate stats, and write summary
        global_step, info["learning_rate"], step_summary = update_stats(
            stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step

            is_overflow = process_stats(
                stats, info, global_step, steps_per_stats)
            print_step_info("  ", global_step, info, 
                "BLEU %.2f" % (hparams.best_bleu, ))
            if is_overflow:
                break

            # Reset statistics
            stats = init_stats()

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step

            utils.log("Save eval, global step %d" % (global_step, ))
            utils.add_summary(summary_writer, global_step, "train_ppl", info["train_ppl"])

            # Save checkpoint
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(model_dir, "translate.ckpt"),
                global_step=global_step)

            # Evaluate on dev/test
            run_sample_decode(infer_model, infer_sess,
                                model_dir, hparams, summary_writer, sample_src_data,
                                sample_tgt_data)
            run_internal_eval(
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
            run_external_eval(
                infer_model, infer_sess, model_dir,
                hparams, summary_writer)
            if avg_ckpts:
                run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                                    summary_writer, global_step)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(model_dir, "translate.ckpt"),
        global_step=global_step)

    (result_summary, _, final_eval_metrics) = run_full_eval(
        model_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data, avg_ckpts)
    print_step_info("Final, ", global_step, info, result_summary)
    utils.log("Done training!")

    summary_writer.close()

    utils.log("Start evaluating saved best models.")
    best_model_dir = hparams.best_bleu_dir
    summary_writer = tf.summary.FileWriter(
        os.path.join(best_model_dir, summary_name), infer_model.graph)
    result_summary, best_global_step, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    print_step_info("Best BLEU, ", best_global_step, info,
                    result_summary)
    summary_writer.close()

    if avg_ckpts:
        best_model_dir = hparams.avg_best_bleu_dir
        summary_writer = tf.summary.FileWriter(
            os.path.join(best_model_dir, summary_name), infer_model.graph)
        result_summary, best_global_step, _ = run_full_eval(
            best_model_dir, infer_model, infer_sess, eval_model, eval_sess,
            hparams, summary_writer, sample_src_data, sample_tgt_data)
        print_step_info("Averaged Best BLEU, ", best_global_step, info,
                        result_summary)
        summary_writer.close()
    
    return final_eval_metrics, global_step

def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data,
                  avg_ckpts=False):
    """
    Wrapper for running sample_decode, internal_eval and external_eval.
    """

    run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                        sample_src_data, sample_tgt_data)
    dev_ppl, test_ppl = run_internal_eval(
        eval_model, eval_sess, model_dir, hparams, summary_writer)
    dev_scores, test_scores, global_step = run_external_eval(
        infer_model, infer_sess, model_dir, hparams, summary_writer)

    metrics = {
        "dev_ppl": dev_ppl,
        "test_ppl": test_ppl,
        "dev_scores": dev_scores,
        "test_scores": test_scores,
    }

    avg_dev_scores, avg_test_scores = None, None
    if avg_ckpts:
        avg_dev_scores, avg_test_scores = run_avg_external_eval(
            infer_model, infer_sess, model_dir, hparams, summary_writer,
            global_step)
        metrics["avg_dev_scores"] = avg_dev_scores
        metrics["avg_test_scores"] = avg_test_scores

    result_summary = utils.format_results("dev", dev_ppl, dev_scores)
    if avg_dev_scores:
        result_summary += ", " + utils.format_results("avg_dev", None, avg_dev_scores)
    if hparams.test_prefix:
        result_summary += ", " + utils.format_results("test", test_ppl, test_scores)
        if avg_test_scores:
            result_summary += ", " + utils.format_results("avg_test", None, avg_test_scores)

    return result_summary, global_step, metrics

def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
    """
    Sample decode a random sentence from src_data.
    """

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

    nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)

    if hparams.beam_width > 0:
        # get the top translation.
        nmt_outputs = nmt_outputs[0]

    translation = utils.get_translation(
        nmt_outputs,
        sent_id=0,
        tgt_eos=hparams.eos)

    utils.log("Sample src: {}".format(src_data[decode_id]))
    utils.log("Sample ref: {}".format(tgt_data[decode_id]))
    utils.log("NMT output: {}".format(translation))

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
    eval_model, eval_sess, model_dir, hparams, summary_writer, use_test_set=True):
    """
    Compute internal evaluation (perplexity) for both dev / test.
    """

    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_model.model, model_dir, eval_sess, "eval")

    utils.log("Internal evaluation, global step %d" % global_step)

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
    if use_test_set and hparams.test_prefix:
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
                   save_on_best, avg_ckpts=False):
    """
    External evaluation such as BLEU and ROUGE scores.
    """

    out_dir = hparams.out_dir  
    if avg_ckpts:
        label = "avg_" + label 

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = model.decode_and_evaluate(
        label,
        sess,
        output,
        ref_file=tgt_file,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos)

    # Save on best metrics
    if avg_ckpts:
        best_metric_label = "avg_best_bleu"
    else:
        best_metric_label = "best_bleu"
    utils.add_summary(summary_writer, global_step, "%s_bleu" % (label, ),
                        scores['BLEU'])
    # metric: larger is better
    if save_on_best and scores['BLEU'] > getattr(hparams, best_metric_label):
        setattr(hparams, best_metric_label, scores['BLEU'])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    
    utils.save_hparams(out_dir, hparams)

    return scores

def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True, use_test_set=True,
                      avg_ckpts=False):
    """
    Compute external evaluation (bleu, rouge, etc.) for both dev / test.
    """

    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    dev_scores = None
    test_scores = None
    if global_step > 0:

        utils.log("External evaluation, global step %d" % global_step)

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
            save_on_best=save_best_dev,
            avg_ckpts=avg_ckpts)

        if use_test_set and hparams.test_prefix:
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
                save_on_best=False,
                avg_ckpts=avg_ckpts)

    return dev_scores, test_scores, global_step

def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                          summary_writer, global_step):
    """
    Creates an averaged checkpoint and run external eval with it.
    """

    avg_dev_scores, avg_test_scores = None, None
    if hparams.avg_ckpts:
        # Convert VariableName:0 to VariableName.
        global_step_name = infer_model.model.global_step.name.split(":")[0]
        avg_model_dir = model_helper.avg_checkpoints(
            model_dir, hparams.num_keep_ckpts, global_step, global_step_name)

        if avg_model_dir:
            avg_dev_scores, avg_test_scores, _ = run_external_eval(
                infer_model,
                infer_sess,
                avg_model_dir,
                hparams,
                summary_writer,
                avg_ckpts=True)

    return avg_dev_scores, avg_test_scores