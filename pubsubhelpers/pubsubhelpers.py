"""
Contains some helper code for dealing with the google
cloud pub/sub api
"""

from __future__ import unicode_literals

import logging
import base64
import sys
import time

from oauth2client import client as oauth2client
from apiclient import discovery
import httplib2
import HTMLParser

def create_default_client(scopes=['https://www.googleapis.com/auth/pubsub']):
    """Attempts to create a default client. The scopes argument may not be required.
    """
    credentials = oauth2client.GoogleCredentials.get_application_default()
    if credentials.create_scoped_required():
        credentials = credentials.create_scoped(scopes)
    http = httplib2.Http()
    credentials.authorize(http)
    return discovery.build('pubsub', 'v1', http=http)

def check_pubsub_subscription(client, sub_name, topic_name, 
                              create=True):
    """double checks subscriptions exists"""

    subs = client.projects().subscriptions().list(
        project="/".join(sub_name.split("/")[:2])).execute()
    logging.debug(subs)
    found = False
    try:
        for sub in subs['subscriptions']:
            if sub['name'] == sub_name:
                found = True
    except KeyError:
        found = False

    if not found:
        if create:
            logging.error('could not find subscription, setting up')
            client.projects().subscriptions().create(
                name=sub_name,
                body={'topic':topic_name}).execute()
        else:
            logging.error('no subscription')
            return False
    else:
        logging.info('subscription ok!')
    return True


def pull_pubsub_messages(client, subscription, topic, tries=1, wait=5):
    """Checks if a subscriptions exists, returns messages from it,
    If no messages are received, tries a couple of times (if desired),
    sometimes they can take a while to come through.
    """
    check_pubsub_subscription(client, subscription, topic)
    batch_size = 100 # who knows how many there will be?
    body = {
        'returnImmediately': True,
        'maxMessages': batch_size
        }
    queueEmpty = False
    msgs = []
    h = HTMLParser.HTMLParser()
    logging.info('checking messages')
    for t in range(tries):
        while not queueEmpty:
            resp = client.projects().subscriptions().pull(
                subscription=subscription, body=body).execute()
            received_messages = resp.get('receivedMessages')
            if received_messages is not None:
                ack_ids = []
                for received_message in received_messages:
                    pubsub_message = received_message.get('message')
                    if pubsub_message:
                        msg = base64.b64decode(pubsub_message.get('data')).decode('utf-8')
                        msg = h.unescape(msg)
                        print(type(msg),msg)
                        msgs.append(msg)
                        ack_ids.append(received_message.get('ackId'))
                ack_body = {'ackIds':ack_ids}
                client.projects().subscriptions().acknowledge(
                    subscription=subscription, body=ack_body).execute()
                queueEmpty = len(msgs) < batch_size
            else:
                queueEmpty = True
        if len(msgs) != 0: # if we've got something, good
            break
        time.sleep(wait)
    # should have some stuff to play with
    logging.info('got {} messages'.format(len(msgs)))
    return msgs

def post_pubsub_messages(topic, messages):
    """Posts messages to topic. Expects messages to be a list of strings
    and assumes topic exists"""
    client = create_default_client()
    # make a body (this is POST)
    body = {
        'messages': [
            {'data': base64.b64encode(m.encode("utf-8"))} for m in messages if len(m) > 0
            ]
        }
    resp = client.projects().topics().publish(
        topic=topic, body=body).execute()
    logging.debug(resp)
    if 'error' in resp:
        logging.error("Had some kind of issue with publishing")
    else:
        logging.info("Tried to publish {} messages to {}, did not check response".format(
            len(messages), topic))
    
