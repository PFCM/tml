#! /usr/bin/python

# checks to see if there are new tweets
# if so, should load them up, clean them and 
# proceed

# if not it should really shut itself down

#from __future__ import unicode_literals

import logging
import datetime
LOGFILE = 'log'
logging.basicConfig(filename=LOGFILE,
                    level=logging.DEBUG)
import base64
import sys

from oauth2client import client as oauth2client
from apiclient import discovery
import httplib2

import pubsubhelpers
import charmodel

import codecs

MODEL_PATH = "./models/"

def train_and_sample(data):
    # if data is none, just gen samples
    conf = charmodel.get_config()
    conf.hidden_size = 512
    conf.num_layers = 1 
    conf.batch_size = 8
    conf.max_max_epoch = 2 if data else 0
    train_data = charmodel.reader.tweets_to_sequence(data) if data else []
    #print(charmodel.reader.get_vocab())
    
    return charmodel.train_and_sample(train_data, 
                                      conf,
                                      MODEL_PATH,
                                      charmodel.reader.get_vocab(),
                                      sample_length=500)

def main():
    logging.info('firing up')
    new_tweets = pubsubhelpers.pull_pubsub_messages(
        pubsubhelpers.create_default_client(),
        'projects/twittest-1140/subscriptions/get_data',
        'projects/twittest-1140/topics/new_data',
        tries=3,
        wait=20)
    if len(new_tweets) > 0:
        tweets = list(u"\n".join(new_tweets))
        #with codecs.open("bootstrap_data.txt", "r", "utf-8", errors='ignore') as f:
        #tweets = list(f.read())
    else:
        tweets = None

    sample = unicode(train_and_sample(tweets))
    print(type(sample))
    samples = sample.split(u"\n")
    # clip them to 140 chars jic
    for i,s in enumerate(samples):
        if len(s) > 140:
            samples[i] =  s[:140]
#    for s in samples:
#        print(s, len(s))
    pubsubhelpers.post_pubsub_messages('projects/twittest-1140/topics/new_tweet',
                                       samples)
    # byeee
    pubsubhelpers.post_pubsub_messages('projects/twittest-1140/topics/turnmeoff',['instance-1'])

if __name__ == '__main__':
    main()
