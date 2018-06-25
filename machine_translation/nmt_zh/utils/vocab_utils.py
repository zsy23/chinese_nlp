# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

import os
import tensorflow as tf

from utils import misc_utils as utils

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
    """
    Check if vocab_file doesn't exist, create from corpus_file.
    """
    
    if os.path.exists(vocab_file):
        utils.log("Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            # Verify if the vocab starts with unk, sos, eos
            # If not, prepend those tokens & generate a new vocab file
            if not unk: unk = UNK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                utils.log("The first 3 vocab words [%s, %s, %s]"
                                " are not [%s, %s, %s]" %
                                (vocab[0], vocab[1], vocab[2], unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with open(new_vocab_file, "w", encoding='utf-8') as f:
                    for word in vocab:
                        f.write("%s\n" % (word, ))
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file '%s' does not exist." % (vocab_file, ))

    vocab_size = len(vocab)
    return vocab_size, vocab_file

def load_vocab(vocab_file):
    vocab = []
    with open(vocab_file, "r", encoding='utf-8') as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
        
    return vocab, vocab_size

def load_embed_txt(embed_file):
    """
    Load embed_file into a python dictionary.
    """
    emb_dict = dict()
    emb_size = None
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """
    Creates vocab tables for src_vocab_file and tgt_vocab_file.
    """

    src_vocab_table = tf.contrib.lookup.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
            tgt_vocab_file, default_value=UNK_ID)
            
    return src_vocab_table, tgt_vocab_table
