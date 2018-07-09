# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the Vocab class for converting string to id and back
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, lower=False):
        self.lower = lower

        self.word_id2token = {}
        self.word_token2id = {}
        self.word_token_cnt = {}
        self.char_id2token = {}
        self.char_token2id = {}    
        self.char_token_cnt = {}    

        self.word_embed_dim = None
        self.word_embeddings = None
        self.char_embed_dim = None
        self.char_embeddings = None

        self.pos_size = None
        self.pos_embed_dim = None
        self.pos_embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<oov>'

        self.initial_tokens = []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token, 'word')
            self.add(token, 'char')


    def size(self, embed_type = 'word'):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        if embed_type == 'word':
            return len(self.word_id2token)
        elif embed_type == 'char':
            return len(self.char_id2token)


    def get_id(self, token, embed_type = 'word'):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            if embed_type == 'word':
                return self.word_token2id[token]
            elif embed_type == 'char':
                return self.char_token2id[token]
        except KeyError:
            if embed_type == 'word':
                return self.word_token2id[self.unk_token]
            elif embed_type == 'char':
                return self.char_token2id[self.unk_token]


    def get_token(self, idx, embed_type = 'word'):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            if embed_type == 'word':
                return self.word_id2token[idx]
            elif embed_type == 'char':
                return self.char_id2token[idx]
        except KeyError:
                return self.unk_token
            

    def add(self, token, cnt=1, embed_type = 'word'):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if embed_type == 'word':
            if token in self.word_token2id:
                idx = self.word_token2id[token]
            else:
                idx = len(self.word_id2token)
                self.word_id2token[idx] = token
                self.word_token2id[token] = idx
            if cnt > 0:
                if token in self.word_token_cnt:
                    self.word_token_cnt[token] += cnt
                else:
                    self.word_token_cnt[token] = cnt
        elif embed_type == 'char':
            if token in self.char_token2id:
                idx = self.char_token2id[token]
            else:
                idx = len(self.char_id2token)
                self.char_id2token[idx] = token
                self.char_token2id[token] = idx
            if cnt > 0:
                if token in self.char_token_cnt:
                    self.char_token_cnt[token] += cnt
                else:
                    self.char_token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt, embed_type = 'word'):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        if embed_type == 'word':
            filtered_tokens = [token for token in self.word_token2id if self.word_token_cnt[token] >= min_cnt]
            # rebuild the token x id map
            self.word_token2id = {}
            self.word_id2token = {}
            for token in self.initial_tokens:
                self.add(token, 0, 'word')
            for token in filtered_tokens:
                self.add(token, 0, 'word')
        elif embed_type == 'char':
            filtered_tokens = [token for token in self.char_token2id if self.char_token_cnt[token] >= min_cnt]
            # rebuild the token x id map
            self.char_token2id = {}
            self.char_id2token = {}
            for token in self.initial_tokens:
                self.add(token, 0, 'char')
            for token in filtered_tokens:
                self.add(token, 0, 'char')

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.pos_embed_dim = embed_dim
        self.pos_size = 32
        self.pos_embeddings = np.random.rand(self.pos_size, embed_dim)
        self.pos_embeddings[0] = np.zeros([embed_dim])
        

    def load_pretrained_word_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            info = list(map(int, fin.readline().strip().split()))
            self.word_embed_dim = info[1]
            for line in fin:
                contents = line.strip().split()
                token = ''.join(contents[:0-self.word_embed_dim]).decode('utf8')
                if token not in self.word_token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[0-self.word_embed_dim:]))

        filtered_tokens = trained_embeddings.keys()
        print('total', len(filtered_tokens), 'words')
        # rebuild the token x id map
        self.word_token2id = {}
        self.word_id2token = {}
        for token in self.initial_tokens:
            self.add(token, 0, 'word')
        for token in filtered_tokens:
            self.add(token, 0, 'word')
        # load embeddings
        self.word_embeddings = np.zeros([self.size('word'), self.word_embed_dim])
        self.word_embeddings[self.get_id(self.unk_token, 'word')] = np.random.rand(1, self.word_embed_dim)
        for token in self.word_token2id.keys():
            if token in trained_embeddings:
                self.word_embeddings[self.get_id(token, 'word')] = trained_embeddings[token]


    def load_pretrained_char_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            info = list(map(int, fin.readline().strip().split()))
            self.char_embed_dim = info[1]
            for line in fin:
                contents = line.strip().split()
                token = ''.join(contents[:0-self.char_embed_dim]).decode('utf8')
                if len(token) > 1:
                    continue
                trained_embeddings[token] = list(map(float, contents[0-self.char_embed_dim:]))

        filtered_tokens = trained_embeddings.keys()
        print('total', len(filtered_tokens), 'chars')
        # rebuild the token x id map
        self.char_token2id = {}
        self.char_id2token = {}
        for token in self.initial_tokens:
            self.add(token, 0, 'char')
        for token in filtered_tokens:
            self.add(token, 0, 'char')
        # load embeddings
        self.char_embeddings = np.zeros([self.size('char'), self.char_embed_dim])
        self.char_embeddings[self.get_id(self.unk_token, 'char')] = np.random.rand(1, self.char_embed_dim)
        for token in self.char_token2id.keys():
            if token in trained_embeddings:
                self.char_embeddings[self.get_id(token, 'char')] = trained_embeddings[token]

    def convert_to_ids(self, tokens, embed_type = 'word'):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        if embed_type == 'word':
            vec = [self.get_id(label, 'word') for label in tokens]
        elif embed_type == 'char':
            vec = []
            for w in tokens:
                vec.append([self.get_id(c, 'char') for c in w])
        return vec

    # def recover_from_ids(self, ids, stop_id=None, embed_type = 'word'):
    #     """
    #     Convert a list of ids to tokens, stop converting if the stop_id is encountered
    #     Args:
    #         ids: a list of ids to convert
    #         stop_id: the stop id, default is None
    #     Returns:
    #         a list of tokens
    #     """
    #     tokens = []
    #     for i in ids:
    #         tokens += [self.get_token(i, embed_type)]
    #         if stop_id is not None and i == stop_id:
    #             break
    #     return tokens
