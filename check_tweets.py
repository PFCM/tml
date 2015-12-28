#! /usr/bin/python

"""Checks the pub/sub topic to see if there are any new tweets, if so
writes them to a new file, deleting an older file if there are too many.
It then loads the model and trains it for a while."""

import logging
import logging.config
import datetime
LOGFILE = 'log'

# logging config
logging.config.dictConfig({
    'version':1,
    'formatters': {
        'default': {
            'format':'%(asctime)s(%(levelname)s) %(filename)s.%(funcName)s:%(message)s',
            'datefmt':'%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class':'logging.StreamHandler',
            'formatter': 'default',
            'level':'DEBUG',
        },
        'file'   : {
            'class':'logging.handlers.RotatingFileHandler',
            'formatter':'default',
            'level':'DEBUG',
            'filename':LOGFILE,
        }
    },
    'root':{
        'level':'DEBUG',
        'handlers':['console', 'file']
    }
})

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
    conf.num_layers = 2
    conf.batch_size = 16
    conf.max_max_epoch = 8 if data else 0
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
        tries=5,
        wait=600)
    if len(new_tweets) > 0:
        tweets = [t.replace("\n", "\t")]
        
        #with codecs.open("bootstrap_data.txt", "r", "utf-8", errors='ignore') as f:
        #tweets = list(f.read())
    else:
        tweets = None

    sample = unicode(train_and_sample(tweets).replace('&amp;', '&'))
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
