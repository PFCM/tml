"""This module contains tests for ``writing.py``
It does not clean up after itself (ie. may leave directories and
files all over the place, if cleaning them up is not required for
later tests)"""

from contextlib import contextmanager
from string import ascii_letters
import random
import os
import shutil
import time

import logging

import writing

logging.basicConfig(level=logging.INFO)

def random_str(length=10):
    """Generates a random string.

    Args:
        length (Optional[int]): the length of the string to generate.
            Defaults to 10.
    Returns:
        string (str): a random string of ASCII letters of length ``length``.
    """
    return ''.join([random.choice(ascii_letters) for _ in xrange(length)])

def fake_data(rows, row_length=140):
    """Generates some fake data.

    Specifically a list of random strings of a certain length.

    Args:
        rows (int): how many rows of data to generate.
        row_length (Optional[int]): the length of each random string. Defaults
            to 140.
    Returns:
        data (list of str): the fake data.
    """
    return [random_str(length=row_length) for _ in xrange(rows)]

@contextmanager
def unused_dir(cleanup=True, create=False):
    """A context manager to get a temporary directory.
    Gives the name of a directory which is guaranteed not to exist
    in the current working directory. May clean up afterwards.

    Args:
        cleanup (Optional[bool]): whether or not to delete the directory
            when this context manager exits.
            If True, deletes the directory and anything in it. Defaults to
            True. If False does nothing to the directory when exiting.
        create (Optional[bool]): whether to create the directory.
    Yields:
        the file name.
    """
    dname = random_str()
    while os.path.exists(dname):
        # ok don't use the same seed millions of times?
        dname = random_str()
    if create:
        os.mkdir(dname)
    yield dname
    # and we are back should we clear it all out?
    if cleanup:
        shutil.rmtree(dname)

###### actual tests ######
def test_write_brand_new():
    """Tests to see if ``write_data_file`` appropriately creates a
    directory and the correct number of files when started fresh.

    Args:
        none
    Raises:
        AssertionError: if either:

        - the directory does not exist when it should.
        - there are the wrong number of files in the directory.

    """
    # get a unique name
    with unused_dir() as dname:
        data = fake_data(10000)
        # now we will call it such that we expect 10 files written
        writing.write_data_file(data,
                                directory=dname,
                                max_files=100, # jic
                                max_lines=1000)
        # and now we can check
        assert(os.path.isdir(dname))
        assert(len(os.listdir(dname)) == 10)

def test_rollover():
    """Tests to make sure that existing files are deleted appropriately
    when there are too many.

    Args:
        none
    Raises:
        AssertionError: if the wrong files or the wrong number of files
            are deleted.
    """
    # the first thing we will do is fill up a test directory
    with unused_dir() as dname:
        data = fake_data(10)
        writing.write_data_file(data,
                                directory=dname,
                                max_files=100,
                                max_lines=1)
        # make sure the timestamps change
        time.sleep(5)
        # now let's replace the first five
        writing.write_data_file(data[:5],
                                directory=dname,
                                max_files=10,
                                max_lines=1,
                                name=None)
        assert len(os.listdir(dname)) == 10, "should be 10 files"
        # just print them for debugging
        print(os.listdir(dname))
        # now we need to check that the first five were correctly
        # replaced
        # there should be five with an earlier timestamp but a later
        # number on the end
        fnames = os.listdir(dname)
        first = [fname for fname in fnames if int(fname[-1]) < 5]
        last = [fname for fname in fnames if not fname in first]
        assert len(first) == 5
        assert len(last) == 5
        # now check that all have the same timestamp
        # and a that first's one comes AFTER
        for fname in first[1:]:
            assert fname[:-1] == first[0][:-1]
        for fname in last[1:]:
            assert fname[:-1] == last[0][:-1]
        assert first[0][:-1] > last[0][:-1]
