"""
Contains some helper code for dealing with the google
cloud pub/sub api
"""

from __future__ import unicode_literals

import logging
import base64
import sys

from oauth2client import client as oauth2client
from apiclient import discovery
import httplib2

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


def pull_pubsub_messages(client, subscription, topic):
    """Checks if a subscriptions exists, returns messages from it"""
    check_pubsub_subscription(client, subscription, topic)
    batch_size = 100 # who knows how many there will be?
    body = {
        'returnImmediately': True,
        'maxMessages': batch_size
        }
    queueEmpty = False
    msgs = []
    logging.info('checking messages')
    while not queueEmpty:
        resp = client.projects().subscriptions().pull(
            subscription=subscription, body=body).execute()
        received_messages = resp.get('receivedMessages')
    
        if received_messages is not None:
            ack_ids = []
            for received_message in received_messages:
                pubsub_message = received_message.get('message')
                if pubsub_message:
                    msgs.append(base64.b64decode(
                                pubsub_message.get('data')))
                    ack_ids.append(received_message.get('ackId'))
            ack_body = {'ackIds':ack_ids}
            client.projects().subscriptions().acknowledge(
                subscription=subscription, body=ack_body).execute()
            queueEmpty = len(msgs) < batch_size
        else:
            queueEmpty = True

    # should have some stuff to play with
    logging.info('got {} messages'.format(len(msgs)))
    return msgs
