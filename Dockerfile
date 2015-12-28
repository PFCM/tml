# just designed to build a docker image with all dependencies.
# by default will just run tests
FROM b.gcr.io/tensorflow/tensorflow

# install what we need from apt
RUN apt-get -y update && \
    apt-get install -y curl

COPY ./ ./*/* /opt/tml/
WORKDIR /opt/tml
# do we have pip? let's see
RUN pip install -r requirements.txt

CMD ["nosetests"]
