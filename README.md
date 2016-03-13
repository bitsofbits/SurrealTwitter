# Generate tweets based on the a corpus of source tweets

*A character level recurrent neural net is trained to generate tweets in the
style of a set of existing tweets. *

Source tweets are assumed to be in a column named 'text' in "source_tweets.csv" and can 
be downloaded from   twitter using a tool such as 
[tweet_dumper](https://gist.github.com/yanofsky/5436496).

To Train:

>>> network.fit(dummy_data, dummy_data)

To generate tweets:

>>> generate_tweets(network, count)

Running from the command line will attempt to load existing network data from 
"network.params" and continue training. When finished training, or interupted,
the current network data is dumped to "network.params_out". If you want to continue
training with this data next time, this file should be moved to "network.params".

Running dump_tweets(network, count, path) will dump `count` tweets to the locations 
specified by path.