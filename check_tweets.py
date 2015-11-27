#! /usr/bin/python

# checks to see if there are new tweets
# if so, should load them up, clean them and 
# proceed

# if not it should really shut itself down

#from __future__ import unicode_literals

import logging
logging.basicConfig(level=logging.DEBUG)
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
    conf = charmodel.get_config()
    conf.hidden_size = 512#test
    conf.num_layers = 1
    conf.batch_size=4
    conf.max_max_epochs = 25
    train_data = charmodel.reader.tweets_to_sequence(data)
    #print(charmodel.reader.get_vocab())
    
    return charmodel.train_and_sample(train_data, 
                                      conf,
                                      MODEL_PATH,
                                      charmodel.reader.get_vocab())

def main():
    #logging.info('firing up')
    #new_tweets = pubsubhelpers.pull_pubsub_messages(
    #    pubsubhelpers.create_default_client(),
    #    'projects/twittest-1140/subscriptions/get_data',
    #    'projects/twittest-1140/topics/new_data')
    with codecs.open("bootstrap_data.txt", "r", "utf-8", errors='ignore') as f:
        tweets = list(f.read())
    

    sample = train_and_sample(tweets)
    print(sample)

if __name__ == '__main__':
    main()
