import tensorflow as tf
from tensorflow.models.rnn import rnn
import numpy as np
from collections import defaultdict
import sys
from random import shuffle
import data_helpers
#
#         Params
#
np.random.seed(10)

in_file = 'data/data-min25.ints'
pad_token = 2
batch_size = 250
word_dim = 50
hidden_dim = 100
learning_rate = .1
lr_decay = .2
num_layers = 1
dropout_prob = .2
max_grad_norm = 100
is_training = True
max_epoch = 1
max_max_epoch = 25

#
#          Data
#


ep_pattern_map = defaultdict(list)
pattern_token_map = defaultdict(list)
seq_len = 20
vocab_size = 0
label_size = 0
data_x = []
data_y = []
with open(in_file) as f:
    for i, line in enumerate(f):
        e1, e2, ep, pattern, tokens, label = line.strip().split('\t')
        ep_pattern_map[ep].append(pattern)
        label = int(label)-1
        label_size = max(label_size, label+1)
        token_list = map(int, tokens.split(' '))
        if len(token_list) <= seq_len:
            if len(token_list) < seq_len:
                token_list += [pad_token] * (seq_len - len(token_list))
            pattern_token_map[pattern].append(token_list)
            vocab_size = max(vocab_size, max(token_list)+1)
            data_x.append(token_list)
            data_y.append(label)


    print(str(i) + ' examples\t'
          + str(len(ep_pattern_map)) + ' entity pairs\t'
          + str(len(pattern_token_map)) + ' patterns\t'
          + str(label_size) + ' labels\t'
          + str(vocab_size) + ' unique tokens')

# shuffle the training data
zipped_data = zip(data_x, data_y)
shuffle(zipped_data)
data_x, data_y = zip(*zipped_data)

# convert data to numpy arrays - labels must be dense one-hot vectors
dense_y = []
for i, j in enumerate(data_y):
    dense_y.append([0]*label_size)
    dense_y[i][j] = 1
data_y = np.array(dense_y)
data_x = np.array(data_x)




#
#       Model stuff
#

input_x = tf.placeholder(tf.int32, [None, seq_len], name="input_x")
input_y = tf.placeholder(tf.float32, [None, label_size], name="input_y")

sx, sy = tf.shape(input_x), tf.shape(input_y)
input_x = tf.Print(input_x, [sx, sy])

lookup_table = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
inputs = tf.nn.embedding_lookup(lookup_table, input_x)
inputs = tf.nn.dropout(inputs, 1 - dropout_prob)
inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, seq_len, inputs)]

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, input_size=word_dim)
if is_training and 1 - dropout_prob < 1:
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1 - dropout_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

outputs, state = rnn.rnn(cell, inputs, initial_state=cell.zero_state(batch_size, tf.float32))
# lstm returns [hiddenstate+cell] -- extact just the hidden state
state = tf.slice(state, [0, 0], [batch_size, hidden_dim])
softmax_w = tf.get_variable("softmax_w", [hidden_dim, label_size])
softmax_b = tf.get_variable("softmax_b", [label_size])

logits = tf.nn.xw_plus_b(state, softmax_w, softmax_b, name="logits")
loss = tf.nn.softmax_cross_entropy_with_logits(logits, input_y)
_cost = tf.reduce_sum(loss) / batch_size

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(_cost, tvars), max_grad_norm)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)

optimizer = tf.train.AdamOptimizer(learning_rate)
# grads_and_vars = optimizer.compute_gradients(loss)
# _train_op = optimizer.apply_gradients(grads_and_vars)
_train_op = optimizer.apply_gradients(zip(grads, tvars))


#
#           Train
#
class MinibatchIterator:
    def __init__(self, data_array, batch_size):
        self.data_array = data_array
        self.num_rows = data_array.shape[0]
        self.batch_size = batch_size
        self.start_idx = 0

    def __iter__(self):
        self.start_idx = 0
        return self

    def next(self):
        if self.start_idx >= self.num_rows:
            raise StopIteration
        else:
            end_idx = min(self.start_idx + self.batch_size, self.num_rows)
            to_return = self.data_array[self.start_idx:end_idx]
            self.start_idx = end_idx
            return to_return


with tf.Graph().as_default() and tf.Session() as session:
    tf.initialize_all_variables().run()
    for i in range(max_max_epoch):
        lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        x_batch, y_batch = MinibatchIterator(data_x, batch_size), MinibatchIterator(data_y, batch_size)
        costs = []
        """Runs the model on the given data."""
        for step, (x, y) in enumerate(zip(x_batch, y_batch)):
            if len(x) == batch_size and int(np.sum(y)) > 0:
                cost, _ = session.run([_cost, _train_op], feed_dict={input_x: x, input_y: y})
                costs.append(cost)
                sys.stdout.write('\r{:4.3f} last cost, {:4.3f} avg cost, {:1.0f} step'.format(cost, reduce(lambda x, y: x + y, costs) / len(costs), step))
                sys.stdout.flush()
            else:
                print('no label!', np.sum(x), np.sum(y))
        print '\n' + str(reduce(lambda x, y: x + y, costs) / len(costs))
