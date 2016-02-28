
# Pulled over form notebook.

# Refactor!!!! XXX

# Get the tweets

import pandas as pd
import html
import codecs

tweets_info = pd.read_csv("realDonaldTrump_tweets.csv", dtype=bytes)

tweets = [html.unescape(x) for x in tweets_info['text']]

# See what we are dealing with
from collections import Counter
import numpy as np
c = Counter()
for twt in tweets:
    c.update(twt)
counts = c.most_common()
print(len(counts))
print(counts[:5])
print(max([len(x) for x in tweets]))

# Map characters to tokens and vice versa. Add special START and STOP
# tokens.

chars_to_tokens = {}
for i, (c, cnt) in enumerate(counts):
    chars_to_tokens[c] = i
chars_to_tokens["START"] = i+1
chars_to_tokens["STOP"] = i+2

tokens_to_chars = [None] * len(chars_to_tokens)
for k, v in chars_to_tokens.items():
    tokens_to_chars[v] = k

SEQ_LENGTH = 20

# Batch Size
BATCH_SIZE = 128

SYMBOL_COUNT = len(tokens_to_chars)
def encode(char):
    vec = np.zeros([SYMBOL_COUNT], dtype='float32')
    vec[ chars_to_tokens[char] ] = 1
    return vec
 
# def gen_data_chunk():
#     """yield a sequence of training data based on the tweets"""
#     while True:
#         twt = np.random.choice(tweets)
#         # Encode the whole tweet since they aren't that long
#         all_chars = ["START"] * SEQ_LENGTH + list(twt) + ["STOP"]
#         # Now yield values for each chunk
#         i = np.random.randint(len(all_chars)-SEQ_LENGTH)
#         chars = all_chars[i:i+SEQ_LENGTH]
#         token = chars_to_tokens[ all_chars[i+SEQ_LENGTH] ]
#         vectors = np.zeros([SEQ_LENGTH, SYMBOL_COUNT], dtype=int)
#         assert len(chars) == SEQ_LENGTH
#         for j, c in enumerate(chars):
#             vectors[j] = encode(c)
#         yield i, token, vectors
 

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
                token = chars_to_tokens[ all_chars[i+SEQ_LENGTH] ]
                vectors = np.zeros([SEQ_LENGTH, SYMBOL_COUNT], dtype=int)
                assert len(chars) == SEQ_LENGTH
                for j, c in enumerate(chars):
                    vectors[j] = encode(c)
                yield i, token, vectors
    
generator = gen_data_chunk()

# Originally from 
# https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import itertools
from collections import OrderedDict

import warnings
# See https://groups.google.com/forum/#!msg/lasagne-users/jtXB62wd7mQ/_whqLPgsIQAJ
warnings.filterwarnings('ignore', 'In the strict mode,.+')



char_to_ix = chars_to_tokens
ix_to_char = tokens_to_chars

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))


# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 128

# Optimization learning rate
LEARNING_RATE = 0.01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 50

# Number of epochs to train the net
NUM_EPOCHS = 20

epoch_size = sum(len(x) for x in tweets)

TOKEN_COUNT = SYMBOL_COUNT

# DECAY = 1

STODGINESS = 2

def gen_data():
    x = np.zeros((BATCH_SIZE,SEQ_LENGTH,TOKEN_COUNT))
    y = np.zeros(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        j, c, v = next(generator)
        x[i] = v
        y[i] = c
    return x, y
    
class Layers(OrderedDict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values()).__getitem__(key)
        elif isinstance(key, slice):
            items = list(self.items()).__getitem__(key)
            return Layers(items)
        else:
            return super(Layers, self).__getitem__(key)

    def keys(self):
        return list(super(Layers, self).keys())

    def values(self):
        return list(super(Layers, self).values())
    
start_vector = np.array([[encode("START")] * SEQ_LENGTH])

class Network():
    
    verbose = True
    
    def __init__(self):
        self.learning_rate = theano.shared(np.float32(LEARNING_RATE))

        l_in = lasagne.layers.InputLayer(shape=(None, SEQ_LENGTH, TOKEN_COUNT))

        l_forward_1 = lasagne.layers.LSTMLayer(
            l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)
        
        l_do_1 = lasagne.layers.DropoutLayer(l_forward_1, p=0.5)

        l_forward_2 = lasagne.layers.LSTMLayer(
            l_do_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)

        l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

        l_do_2 = lasagne.layers.DropoutLayer(l_forward_slice, p=0.5)
        
        l_out = lasagne.layers.DenseLayer(l_do_2, 
                                          num_units=SYMBOL_COUNT, 
                                          W = lasagne.init.Normal(), 
                                          nonlinearity=lasagne.nonlinearities.softmax)




        self.set_layers_([l_in, l_forward_1, l_do_1, l_forward_2, 
                          l_forward_slice, l_do_2, l_out])

        
        
        target_values = T.ivector('target_output')

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(l_out)

        network_output_p = lasagne.layers.get_output(l_out, deterministic=True)
        # The loss function is calculated as the mean of the (categorical) 
        # cross-entropy between the prediction and target.
        cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(l_out,trainable=True)

        # Compute AdaGrad updates for training
        print("Computing updates ...")
        updates = lasagne.updates.adagrad(cost, all_params, self.learning_rate)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        self.train = theano.function([l_in.input_var, target_values], cost, 
                                updates=updates, allow_input_downcast=True)
        self.probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

    def initialize(self):
        pass
        
    def set_layers_(self, layers):
        self.layers_ = Layers()
        names = "l_in l_forward_1 l_do_1 l_forward_2 l_forward_slice l_do_2 l_out".split()
        for n, l in zip(names, layers):
            self.layers_[n] = l

    # From nolearn
    
    def get_all_params_values(self):
        return_value = OrderedDict()
        for name, layer in self.layers_.items():
            return_value[name] = [p.get_value() for p in layer.get_params()]
        return return_value

    def load_params_from(self, source):
        self.initialize()

        if isinstance(source, str):
            with open(source, 'rb') as f:
                source = pickle.load(f)

#         if isinstance(source, NeuralNet):
#             source = source.get_all_params_values()

        success = "Loaded parameters to layer '{}' (shape {})."
        failure = ("Could not load parameters to layer '{}' because "
                   "shapes did not match: {} vs {}.")

        for key, values in source.items():
            layer = self.layers_.get(key)
            if layer is not None:
                for p1, p2v in zip(layer.get_params(), values):
                    shape1 = p1.get_value().shape
                    shape2 = p2v.shape
                    shape1s = 'x'.join(map(str, shape1))
                    shape2s = 'x'.join(map(str, shape2))
                    if shape1 == shape2:
                        p1.set_value(p2v)
                        if self.verbose:
                            print(success.format(
                                key, shape1s, shape2s))
                    else:
                        if self.verbose:
                            print(failure.format(
                                key, shape1s, shape2s))

    def save_params_to(self, fname):
        params = self.get_all_params_values()
        with open(fname, 'wb') as f:
            pickle.dump(params, f, -1)
   
TWITTER_LIMIT = 140

# XXX could generate a bunch of tweets in parallel. Just drop stops and keep going.
def generate_tweet(network, N=144, min_length=64):
    sample_ix = []
    x = start_vector
    for i in range(TWITTER_LIMIT):
        # Sample from the distribution instead:
        p = network.probs(x).ravel()
        if i:
            # After the first symbol we increase the probability of getting the most
            # likely candidates by raising p to STODGINESS. Makes tweets more boring
            # but more comprehensisble (in theory at least)
            p **= STODGINESS
            p /= p.sum()
        ix = np.random.choice(np.arange(SYMBOL_COUNT), p=p)
        if tokens_to_chars[ix] == "STOP":
            # Don't allow short tweets
            if i >= min_length:
                break
        else:
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

    return ''.join(ix_to_char[ix] for ix in sample_ix)    

def run(num_epochs=NUM_EPOCHS, network=None):
    
    if network is None:
        network = Network()
        




    
    print("Training ...")
    try:
        for it in itertools.count():
            
            twt = generate_tweet(network) # Generate text using the p^th character as the start. 
            print("----\n %s \n----" % twt)


            total_cost = 0;
            for _ in range(PRINT_FREQ):
                x, y = gen_data()
                total_cost += network.train(x, y)
            epochs = it * BATCH_SIZE * PRINT_FREQ / epoch_size
            print("Epoch {0} average loss = {1}".format(epochs, total_cost / PRINT_FREQ))
            if epochs > NUM_EPOCHS:
                break
#             network.learning_rate.set_value(np.float32(DECAY * 
#                                                 network.learning_rate.get_value()))
            
    except KeyboardInterrupt:
        pass
    return network



if __name__ == '__main__':
    network = Network()
    try:
        network.load_params_from("network.params_in")
    except:
        print("couldn't load old params")
    try:
        run(network=network)
    except KeyboardInterrupt:
        pass    
    network.save_params_to("network.params_out")
    