import random
import collections
import pandas as pd
import tensorflow as tf

train_data = []
valid_data = []
infer_data = []

def get_embed_vocab(embed_file):
    vocab = {}
    embed = []
    embed_size = -1
    vocab_size = 1
    with open(embed_file, 'r') as fin:
        line = fin.readline().strip().split(' ')
        if len(line) == 2:
            embed_size = int(line[1])
            embed.append([0.0] * embed_size)
        else:
            vocab[line[0]] = vocab_size
            vocab_size += 1
            embed_size = len(line) - 1
            embed.append([0.0] * embed_size)
            embed.append(list(map(float, line[1:])))
        for line in fin:
            line = line.strip().split(' ')
            assert embed_size + 1 == len(line), 'All embedding size should be the same'
            vocab[line[0]] = vocab_size
            vocab_size += 1
            embed.append(list(map(float, line[1:])))

    return embed, vocab


def init_data(mode, vocab, question_file, train_file=None, valid_file=None, infer_file=None, test=False):
    global train_data, valid_data, infer_data

    questions = pd.read_csv(question_file)
    if mode == 'word':
        seq = questions['words']
    elif mode == 'char':
        seq = questions['chars']
    if not test:
        content = pd.read_csv(train_file)
        train_data = []
        for item in zip(content['label'], content['q1'], content['q2']):
            q1_words = seq[int(item[1][1:])].split(' ')
            q2_words = seq[int(item[2][1:])].split(' ')

            one = []
            one.append(item[0])
            one.append([vocab[w] for w in q1_words])
            one.append([vocab[w] for w in q2_words])
            train_data.append(one)
            one = []
            one.append(item[0])
            one.append([vocab[w] for w in q2_words])
            one.append([vocab[w] for w in q1_words])
            train_data.append(one)
        
        content = pd.read_csv(valid_file)
        valid_data = []
        for item in zip(content['label'], content['q1'], content['q2']):
            one = []
            q1_words = seq[int(item[1][1:])].split(' ')
            q2_words = seq[int(item[2][1:])].split(' ')

            one.append(item[0])
            one.append([vocab[w] for w in q1_words])
            one.append([vocab[w] for w in q2_words])
            valid_data.append(one)

    else:
        content = pd.read_csv(infer_file)
        infer_data = []
        for item in zip(content['q1'], content['q2']):
            one = []
            q1_words = seq[int(item[0][1:])].split(' ')
            q2_words = seq[int(item[1][1:])].split(' ')

            one.append(0)
            one.append([vocab[w] for w in q1_words])
            one.append([vocab[w] for w in q2_words])
            infer_data.append(one)


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "labels", "q1", "q2", "q1_len", "q2_len"))):
    pass


def get_iterator(mode, batch_size, shuffle=False):
    if mode == 'train':
        global train_data
        data = train_data
    elif mode == 'valid':
        global valid_data
        data = valid_data
    elif mode == 'infer':
        global infer_data
        data = infer_data

    def gen():
        for one in data:
            yield (one[0], one[1], one[2])

    dataset = tf.data.Dataset.from_generator(
        gen, 
        (tf.int32, tf.int32, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 1000)
    dataset = dataset.map(
        lambda label, q1, q2: 
        (label, q1, q2, tf.size(q1), tf.size(q2)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([]),  # label
                tf.TensorShape([None]),  # q1
                tf.TensorShape([None]),  # q2
                tf.TensorShape([]),  # q1_len
                tf.TensorShape([])),  # q2_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                0,  # label -- unused
                0,  # q1
                0,  # q2
                0,  # q1_len -- unused
                0))  # q2_len -- unused

    batched_dataset = batching_func(dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    (labels, q1, q2, q1_len, q2_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        labels=labels,
        q1=q1,
        q2=q2,
        q1_len=q1_len,
        q2_len=q2_len)
