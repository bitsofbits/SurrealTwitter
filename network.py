# Originally from 
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, LSTMLayer, DropoutLayer, DenseLayer
from lasagne.nonlinearities import tanh, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adagrad
import pickle
import itertools
from collections import OrderedDict
import pandas as pd
import html
import codecs
from collections import Counter
import numpy as np
import warnings
from nolearn.lasagne import NeuralNet, TrainSplit, BatchIterator
# See https://groups.google.com/forum/#!msg/lasagne-users/jtXB62wd7mQ/_whqLPgsIQAJ
warnings.filterwarnings('ignore', 'In the strict mode,.+')
# Because we are using no validation data, we get the following runtime warning.
warnings.filterwarnings('ignore', 'Mean of empty slice.')

SEQ_LENGTH = 50

BATCH_SIZE = 128

TWITTER_LIMIT = 140



def read_tweets(path):
    tweets_info = pd.read_csv(path, dtype=bytes)
    return [html.unescape(x) for x in tweets_info['text']]

def make_token_maps(tweets):
    c = Counter()
    for twt in tweets:
        c.update(twt)
    counts = c.most_common()
    # Map characters to tokens and vice versa. Add special START and STOP
    # tokens.
    char_to_token = {}
    for i, (c, cnt) in enumerate(counts):
        char_to_token[c] = i
    char_to_token["START"] = i+1
    char_to_token["STOP"] = i+2
    #
    token_to_char = [None] * len(char_to_token)
    for k, v in char_to_token.items():
        token_to_char[v] = k
    return token_to_char, char_to_token
    
tweets = read_tweets("realDonaldTrump_tweets.csv")
token_to_char, char_to_token = make_token_maps(tweets)

symbol_count = len(token_to_char)



def encode(char):
    vec = np.zeros([symbol_count], dtype='float32')
    vec[ char_to_token[char] ] = 1
    return vec
    
 
def gen_data_chunk(seed=1234):
    """yield a sequence of training data based on the tweets"""
    np.random.seed(seed)
    order = np.arange(len(tweets))
    while True:
        np.random.shuffle(order)
        for ndx in order:
            twt = tweets[ndx]
            all_chars = ["START"] * SEQ_LENGTH + list(twt) + ["STOP"]
            # Now yield values for each chunk
            for i in range(len(all_chars)-SEQ_LENGTH):
                chars = all_chars[i:i+SEQ_LENGTH]
                token = char_to_token[ all_chars[i+SEQ_LENGTH] ]
                vectors = np.zeros([SEQ_LENGTH, symbol_count], dtype=int)
                assert len(chars) == SEQ_LENGTH
                for j, c in enumerate(chars):
                    vectors[j] = encode(c)
                yield i, token, vectors
    
generator = gen_data_chunk()





# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = 0.01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 100

# Number of epochs to train the net
NUM_EPOCHS = 200
epoch_size = sum(len(x) for x in tweets)
chunks_per_epoch = int(round(epoch_size / PRINT_FREQ))


token_count = symbol_count + 1

STODGINESS = 2
FOOTLOOSE_LENGTH = 20

 
    
def gen_data(n):
    x = np.zeros((n,SEQ_LENGTH,token_count), dtype='float32')
    y = np.zeros(n, dtype='int32')
    for i in range(n):
        j, c, v = next(generator)
        x[i,:,:symbol_count] = v
        where_in_tweet = j / float(TWITTER_LIMIT - 1)
        x[i,:,-1] = where_in_tweet
        y[i] = c
    return x, y
    
    
class MyBatchIterator(BatchIterator):
    def transform(self, X_indices, y_indices):
        n = len(X_indices)
        x, y  = gen_data(n)
        if y_indices is None:
            y = None
        return x, y
    
class OnEpochFinished:
    def __call__(self, nn, train_history):
        twt = generate_tweet(network) 
        print("==>", twt)



def make_network():
    
    learning_rate = theano.shared(np.float32(LEARNING_RATE))

    args = dict
    layers = [(InputLayer, args(name="l_in", shape=(None, SEQ_LENGTH, token_count))),
              (LSTMLayer, args(name="l_forward_1", num_units=N_HIDDEN, grad_clipping=GRAD_CLIP, 
                           nonlinearity=tanh)),
              (DropoutLayer, args(name="l_do_1", p=0.5)),
              (LSTMLayer, args(name="l_forward_2", num_units=N_HIDDEN, grad_clipping=GRAD_CLIP,
                            nonlinearity=tanh, only_return_final=True)),
              (DropoutLayer, args(name="l_do_2", p=0.5)),
              (DenseLayer, args(name="l_out", num_units=symbol_count, W=lasagne.init.Normal(),
                             nonlinearity=softmax))]
    return NeuralNet(
        y_tensor_type = T.ivector,
        layers = layers,
        batch_iterator_train=MyBatchIterator(batch_size=PRINT_FREQ),
        max_epochs=NUM_EPOCHS * chunks_per_epoch,
        verbose=1,
        train_split=TrainSplit(0),
        objective_loss_function = categorical_crossentropy,
        update = adagrad, 
        update_learning_rate = learning_rate,
        on_epoch_finished=[OnEpochFinished()],
    )
    


start_vector = np.array([[np.concatenate([encode("START"), [0]])] * SEQ_LENGTH], dtype='float32')

# XXX could generate a bunch of tweets in parallel. Just drop stops and keep going.
def generate_tweet(network, N=144, min_length=64):
    chars = []
    x = start_vector.copy()
    for i in range(TWITTER_LIMIT):
        # Sample from the distribution instead:
        p = network.predict_proba(x).ravel()
        if i <= FOOTLOOSE_LENGTH:
            # After the first symbol we increase the probability of getting the most
            # likely candidates by raising p to STODGINESS. Makes tweets more boring
            # but more comprehensible (in theory at least)
            p **= STODGINESS
            p /= p.sum()
            tkn = np.random.choice(np.arange(symbol_count), p=p)
        else:
            tkn = np.argmax(p)
        if token_to_char[tkn] == "STOP":
            # Don't allow short tweets
            if i >= min_length:
                break
        else:
            char = token_to_char[tkn]
            chars.append(char)
            x[0,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[0,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1, tkn] = 1. 
            x[0,SEQ_LENGTH-1,-1] = i / float(TWITTER_LIMIT - 1)
    return ''.join(chars)   



def load_params(network, path):
    try:
        network.load_params_from(path)
    except:
        print("couldn't load old params")

def dump_tweets(path, network, param_path, number=128):
    load_params(network, path=param_path)
    file = open(path, 'w')
    for _ in range(number):
        twt = generate_tweet(network).replace('\n', ' ')
        file.write(twt+'\n')

if __name__ == '__main__':
    network = make_network()
    load_params(network, path="network2.params_in")
    try:
        dummy_data = [None] * PRINT_FREQ * BATCH_SIZE
        network.fit(dummy_data, dummy_data)
    except KeyboardInterrupt:
        pass    
    network.save_params_to("network2.params_out")
    