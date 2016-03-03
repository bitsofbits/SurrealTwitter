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
import pandas as pd
import html
import codecs
from collections import Counter
import warnings
from nolearn.lasagne import NeuralNet, TrainSplit, BatchIterator
# See https://groups.google.com/forum/#!msg/lasagne-users/jtXB62wd7mQ/_whqLPgsIQAJ
warnings.filterwarnings('ignore', 'In the strict mode,.+')
# Because we are using no validation data, we get the following runtime warning.
warnings.filterwarnings('ignore', 'Mean of empty slice.')


#
# Read the tweets and tokenize them
#

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
    return np.array(token_to_char), char_to_token
    
tweets = read_tweets("realDonaldTrump_tweets.csv")
token_to_char, char_to_token = make_token_maps(tweets)

symbol_count = len(token_to_char)

#
# Define a generator and will yield a sequence of randomly selected training vectors
#

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
    

#
# Constants affecting how the net is built and trained and tweets generated.
#

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# How long a sequence to train the net with.
SEQ_LENGTH = 50

# Batch size to use. Larger values will take less time per epoch, but will also likely
# take more epochs to train. Large enough values will also overwhelm your GPU.
BATCH_SIZE = 128

# The length of a tweet.
TWITTER_LIMIT = 140

# By default, don't generate tweets shorter than this.
MIN_LENGTH = 64

# Learning rate for neural net.
LEARNING_RATE = 0.01

# All gradients above this will be clipped in LSTM
GRAD_CLIP = 100

# Instead of training on epochs we train on sub epoch chunks. CHUNK_SIZE is how
# many batches are in one chunk. This allows us to check our training more
# frequently
CHUNK_SIZE = 100

# Number of epochs to train the net
MAX_EPOCHS = 200

# STODGINESS and RANDOM_LENGTH control how random the generation of tweets is. 
# For the first RANDOM_LENGTH characters the characters a chosen randomly with
# probability proportional to p**STODGINESS, where p is the raw probability 
# predicted by the net. After RANDOM_LENGTH characters, the most likely 
# character is chosen and the process becomes deterministic.
STODGINESS = 2

RANDOM_LENGTH = 20

epoch_size = sum(len(x) for x in tweets)
chunks_per_epoch = epoch_size / (CHUNK_SIZE * BATCH_SIZE)

# `token_count` is the number of tokens input to the net. There is one more token than
# symbol since we also pass in a value that indicates where in the tweet we are since
# tweet structure varies by location and I want to give the net a chance to learn how
# to end the tweet cleanly. (XXX in retrospect, token may not be the best terminology,
# consider renaming).
token_count = symbol_count + 1


#
# Build the net.
#

class MyBatchIterator(BatchIterator):
    def __init__(self, **kwargs):
        super(MyBatchIterator, self).__init__(**kwargs)
        self.generator = gen_data_chunk()
    
    def gen_data(self, n):
        x = np.zeros((n,SEQ_LENGTH,token_count), dtype='float32')
        y = np.zeros(n, dtype='int32')
        for i in range(n):
            j, c, v = next(self.generator)
            x[i,:,:symbol_count] = v
            where_in_tweet = j / float(TWITTER_LIMIT - 1)
            x[i,:,-1] = where_in_tweet
            y[i] = c
        return x, y
        
    def transform(self, X_indices, y_indices):
        n = len(X_indices)
        x, y  = self.gen_data(n)
        if y_indices is None:
            y = None
        return x, y
    
class OnEpochFinished:
    def __call__(self, nn, train_history):
        twt = generate_tweet(nn) 
        chunk = train_history[-1]['epoch']
        print("{0}: ==> {1}".format(chunk / chunks_per_epoch, twt))


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
        batch_iterator_train=MyBatchIterator(batch_size=CHUNK_SIZE),
        max_epochs=int(round(MAX_EPOCHS * chunks_per_epoch)),
        verbose=1,
        train_split=TrainSplit(0),
        objective_loss_function = categorical_crossentropy,
        update = adagrad, 
        update_learning_rate = learning_rate,
        on_epoch_finished=[OnEpochFinished()],
    )
    



def generate_tweet(network, min_length=MIN_LENGTH):
    chars = []
    x = np.array([[np.concatenate([encode("START"), [0]])] * SEQ_LENGTH], dtype='float32')
    for i in range(TWITTER_LIMIT):
        p = network.predict_proba(x).ravel()
        if i <= RANDOM_LENGTH:
            # Increase the probability of getting the most
            # likely candidates by raising p to STODGINESS. Makes tweets more boring
            # but more comprehensible (in theory at least)
            p **= STODGINESS
            p /= p.sum()
            tkn = np.random.choice(np.arange(symbol_count), p=p)
        else:
            # After RANDOM_LENGTH characters, just use the most likely character.
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

# XXX could generate a bunch of tweets in parallel. Just drop stops and keep going.
def generate_tweets(network, count, min_length=MIN_LENGTH):
    chars = []
    x = np.array([[np.concatenate([encode("START"), [0]])] * SEQ_LENGTH]*count, dtype='float32')
    for i in range(TWITTER_LIMIT):
        p = network.predict_proba(x)
        if i <= RANDOM_LENGTH:
            # Increase the probability of getting the most
            # likely candidates by raising p to STODGINESS. Makes tweets more boring
            # but more comprehensible (in theory at least)
            p **= STODGINESS
            p /= p.sum(axis=1, keepdims=True)
            tkns = np.random.choice(np.arange(symbol_count), p=p, axis=1)
        else:
            # After RANDOM_LENGTH characters, just use the most likely character.
            tkns = np.argmax(p, axis=1)
        if i < min_length:
            # Convert STOP tokens to spaces
            # XXX need to use comprehension here since tkns_to_char is list or convert
            tkns[token_to_char[tkns] == "STOP"] = char_to_token[' ']
        char = token_to_char[tkn]
        chars.append(char)
        x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
        x[:,SEQ_LENGTH-1,:] = 0
        x[:,SEQ_LENGTH-1, tkn] = 1. 
        x[:,SEQ_LENGTH-1,-1] = i / float(TWITTER_LIMIT - 1)
    return [''.join(x) for x in transpose(chars)]

def dump_tweets(network, count, path, param_path="network.params"):
    load_params(network, path=param_path)
    file = open(path, 'w')
    for _ in range(number):
        twt = generate_tweet(network).replace('\n', ' ')
        file.write(twt+'\n')

if __name__ == '__main__':
    network = make_network()
    try:
        network.load_params_from("network.params")
    except:
        print("could not load network.params")
    try:
        dummy_data = [None] * CHUNK_SIZE * BATCH_SIZE
        network.fit(dummy_data, dummy_data)
    except KeyboardInterrupt:
        pass    
    # Save network params when done, but to a different file to help avoid accidentally 
    # clobering network.params
    network.save_params_to("network.params_out")
    