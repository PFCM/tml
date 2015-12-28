[![Build Status](https://travis-ci.org/PFCM/tml.svg?branch=dataModel)](https://travis-ci.org/PFCM/tml)
# tml
Designed to run on Google Compute Engine. The idea is it spins up, pulls some
messages from a Google Cloud Pub/Sub topic, loads up an RNN model, trains a bit
on the new data and then generates a sequence or two. These are then published
to a difference Pub/Sub topic for consumption elsewhere.
