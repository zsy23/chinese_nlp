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

import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

def create_vocab_tables(src_vocab_file, tgt_vocab_file):

    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""

    src_vocab_table = tf.contrib.lookup.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
    	tgt_vocab_file, default_value=UNK_ID)

    with open(src_vocab_file, 'r', encoding='utf-8') as fin:
        src_vocab_size = len(fin.readlines())

    with open(tgt_vocab_file, 'r', encoding='utf-8') as fin:
        tgt_vocab_size = len(fin.readlines())
	
    return src_vocab_table, tgt_vocab_table, src_vocab_size, tgt_vocab_size
