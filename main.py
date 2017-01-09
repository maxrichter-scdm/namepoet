import tensorflow as tf
import tensorlayer as tl
import numpy as np
from os import listdir
from os.path import isfile, join
import random

training = False

# Alphabet size. The characters can range in ASCII values from
# 0 to 98
ALPHASIZE = 98
CELLSIZE = 512
NLAYERS = 3
SEQLEN = 30
BATCHSIZE = 10

# Xd Shape: [BATCHSIZE, SEQLENGTH]
# Input goes into Xd. It takes in a list of integer lists, where
# each integer represents an ASCII character
Xd = tf.placeholder(tf.uint8, [None, None])
# X Shape: [BATCHSIZE, SEQLENGTH, ALPHASIZE]
# Converts Xd into a one hot encoded vector
X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)
# Yd_ Shape: [BATCHSIZE, SEQLENGTH]
# Desired output
Yd_ = tf.placeholder(tf.uint8, [None, None])
# Y_ Shape: [BATCHSIZE, SEQLENGTH, ALPHASIZE]
# Convert Yd_ into one hot form
Y_ = tf.one_hot(Yd_, ALPHASIZE, 1.0, 0.0)
# Hin shape: [BATCHSIZE, CELLSIZE * NLAYERS]
# Input to recurrent layers
Hin = tf.placeholder(tf.float32, [None, CELLSIZE*NLAYERS])

# the model
cell = tf.nn.rnn_cell.GRUCell(CELLSIZE)
mcell = tf.nn.rnn_cell.MultiRNNCell([cell]*NLAYERS, state_is_tuple=False)

Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

# softmax output layer
Hf = tf.reshape(Hr, [-1, CELLSIZE])
Ylogits = tf.contrib.layers.linear(Hf, ALPHASIZE)
Y = tf.nn.softmax(Ylogits)
Yp = tf.argmax(Y, 1)
Yp = tf.reshape(Yp, [BATCHSIZE, -1])

# loss and training step (optimizer)
loss = tf.nn.softmax_cross_entropy_with_logits(Y, Y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

if training:
    # Load up all the training files
    path = "corpus/"
    files = [open(join(path, f)) for f in listdir(path) if isfile(join(path, f))]
    text_list = [f.read() for f in files]
    # Join them all together with a newline
    text = "\n".join(text_list)
    [f.close() for f in files]
    # Convert the text into the ASCII equivalents
    char_list = map(ord, list(text))
    for epoch in range(20):
        inH = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])
        for x, y_ in tl.iterate.ptb_iterator(char_list, batch_size=BATCHSIZE, num_steps=SEQLEN):
            dic = {Xd : x, Yd_ : y_, Hin : inH}
            _,y,outH = sess.run([train_step, Yp, H], feed_dict=dic)
            inH = outH
        saver.save(sess, "name-poet-%i" % epoch)
else:
    new_saver = tf.train.import_meta_graph('name-poet-19.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    x = [[random.randint(ord('A'), ord('z')) for i in xrange(SEQLEN)] for j in xrange(BATCHSIZE)]
    inH = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])
    output = ""
    for char_i in xrange(10000):
        dic = {Xd : x, Hin : inH}
        y, outH = sess.run([Yp, H], feed_dict=dic)
        x = y
        inH = outH
        print y
