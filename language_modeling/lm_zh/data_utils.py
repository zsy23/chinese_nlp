import os, math
import logging
import numpy as np
import tensorflow as tf
from collections import Counter

class Vocabulary(object):

    def __init__(self):
        self.token2id = {}
        self.id2token = []
        self.vocab_size = 0
        self.pad = '<pad>'
        self.pad_id = None
        self.unk = '<unk>'
        self.unk_id = None
        self.start = '<s>'
        self.start_id = None
        self.end = '</s>'
        self.end_id = None
        self.n = 'N'
        self.n_id = None

    def add(self, token):
        self.token2id[token] = self.vocab_size
        self.id2token.append(token)
        self.vocab_size += 1

    def finalize(self):
        self.pad_id = self.get_id(self.pad)
        self.unk_id = self.get_id(self.unk)
        self.start_id = self.get_id(self.start)
        self.end_id = self.get_id(self.end)
        self.n_id = self.get_id(self.n)

    def get_id(self, token):
        return self.token2id.get(token, self.unk_id)

    def get_token(self, id_):
        return self.id2token[id_]

    @staticmethod
    def from_file(filename):
        vocab = Vocabulary()
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                vocab.add(word)
        vocab.finalize()
        return vocab

class Dataset(object):

    def __init__(self, prefix, num_steps, batch_size, train=True):
        self.prefix = prefix
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.data_types = None
        self.data_shapes = None
        self.logger = logging.getLogger('lm_zh')

        if os.path.exists(prefix + '.vocab'):
            self.vocab = Vocabulary().from_file(prefix + '.vocab')
        else:
            self.build_vocab([prefix + '.train.txt'])
            self.logger.info('Save vocab to file {}'.format(self.prefix + '.vocab'))
        self.logger.info('Vocab size: {}'.format(self.vocab.vocab_size))

        if train:
            self.train_data, self.train_size = self.build_data(prefix + '.train.txt')
            self.logger.info('Train dataset has {} words'.format(self.train_size))
            self.valid_data, self.valid_size = self.build_data(prefix + '.valid.txt')
            self.logger.info('Valid dataset has {} words'.format(self.valid_size))

            self.data_types = self.train_data.output_types
            self.data_shapes = self.train_data.output_shapes
        else:
            self.test_data, self.test_size = self.build_data(prefix + '.test.txt')
            self.logger.info('Test dataset has {} words'.format(self.test_size))

            self.data_types = self.test_data.output_types
            self.data_shapes = self.test_data.output_shapes

    def build_vocab(self, fn_list):
        words = Counter()
        for fn in fn_list:
            with open(fn, 'r', encoding='utf-8') as fin:
                for line in fin:
                    words.update(line.strip().split(' '))
        self.vocab = Vocabulary()
        self.vocab.add('<pad>')
        self.vocab.add('<unk>')
        self.vocab.add('<s>')
        self.vocab.add('</s>')
        self.vocab.add('N')
        with open(self.prefix + '.vocab', 'w', encoding='utf-8') as fout:
            fout.write('<pad>\n<unk>\n<s>\n</s>\nN\n')
            for k, _ in words.most_common():
                if k in ['<pad>', '<unk>', 'N', '<s>', '</s>']:
                    continue
                self.vocab.add(k)
                fout.write(k + '\n')
        self.vocab.finalize()

    def build_data(self, filename):
        vocab = self.vocab
        data = []       
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                data += [vocab.start_id] + [vocab.get_id(w) for w in line.strip().split(' ')] + [vocab.end_id]

        x = data[:-1]
        w = [1] * len(x)
        x = x + [vocab.pad_id] * (math.ceil((len(x) / self.num_steps)) * self.num_steps - len(x))
        w = w + [0] * (len(x) - len(w)) 
        y = data[1:]
        y = y + [vocab.pad_id] * (math.ceil((len(y) / self.num_steps)) * self.num_steps - len(y))
        nnn = len(x) // self.num_steps // self.batch_size * self.batch_size * self.num_steps
        x = x[:nnn]
        w = w[:nnn]
        y = y[:nnn]

        data = tf.data.Dataset.from_tensor_slices(
                (np.array(x, dtype=np.int32).reshape([-1, self.num_steps]),
                np.array(y, dtype=np.int32).reshape([-1, self.num_steps]),
                np.array(w, dtype=np.int32).reshape([-1, self.num_steps])))
        data = data.shuffle(10000)
        data = data.batch(self.batch_size)
        
        return data, len(x)









    
    

