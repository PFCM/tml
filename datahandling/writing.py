"""Helper functions for managing the data files on disk"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import exists
from math import ceil
import datetime
import itertools
import codecs
import logging

def _iter_chunk(iterable, chunksize):
    """Generator to iterate a sequence (or indeed, any iterator) in
    chunks of a specified size. The last chunk may be smaller.

    Args:
        iter (iterator): the iterator or sequence to roll over.
        chunksize (int): the size of the biggest chunk/slice to return.
    Yields:
        slices of at most chunksize (and only the last one may be smaller).
    """
    it = iter(iterable)
    chunk = list(itertools.islice(it, chunksize))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, chunksize))

def _get_name(base=None, num=-1):
    """Gets an appropriate name for a data file.

    Args:
        base (Optional[str]): the base name to append to.
        it (Optional[int]): the current iteration.
    Returns:
        str - a hopefully unique name.
    """
    if base:
        new_name = base
    else:
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if num > -1:
        new_name += '_{}'.format(num)
    return new_name

def write_data_file(data, directory="data", max_files=10, max_lines=5000,
                    name=None):
    """Writes a new data file into the data directory, deleting the oldest
    file if necessary.

    Args:
        data (sequence of data items): the set of data to write to the file.
            For now this is pretty much assumed to be a sequence of strings.
        directory (Optional[str]): the path to the directory containing the
            data files. The directory is created if it does not exist, if it
            does exist all files within are assumed to be data files and some
            number of the oldest ones may be deleted if necessary. Defaults to
            "data".
        max_files (Optional[int]): the maximum number of files to allow in the
            data directory. Defaults to 10.
        max_lines (Optional[int]): the maximum number of data items to write to a
            file. If ``data`` contains more lines than this, it will be split
            across as many files as necessary. Defaults to 5000.
        name (Optional[str]): the name of the file to write to. If more than one
            file is required (due to the ``max_lines`` parameter) then an
            appropriate number will be appended to the name. If ``None`` then names
            will be chosen based on the current date and time. Default is ``None``.

    Raises:
        IOError: if there is a problem with any of the files or directories.
        ValueError: if the number of files required to write the data file is
            greater than the specified maximum.
    """
    # first see how many files we need
    num_files = int(ceil(len(data) / max_lines))
    if num_files > max_files:
        raise ValueError(
            "Can't write data, required number of files ({}) is greater than max({})".format(
                num_files, max_files
            ))
    # now check if directory exists
    if exists(directory):
        logging.info("Directory '%s' exists, checking number of files...", directory)
        fnames = os.listdir(directory)
        if len(fnames) + num_files > max_files:
            # someone's getting the chop
            num_excess = len(fnames) + num_files - max_files
            logging.info('too many files (%s) geting rid of %s',
                len(fnames), num_excess
            )
            # account for working directories
            fnames = [os.path.join(directory, name) for name in fnames]
            # sort by modified time
            fnames.sort(key=os.path.getmtime)
            # and give the first num_excess the heave-ho
            for old in fnames[:num_excess]:
                os.remove(old)
    else:
        # directory does not exist
        logging.info("Directory '%s' does not exist, creating...", directory)
        os.mkdir(directory)
    # now we definitely have somewhere to write, let's get to it
    # the first thing we have to do is iterate the data in chunks
    for i, data_slice in enumerate(_iter_chunk(data, max_lines)):
        fname = _get_name(base=None, num=i if num_files > 1 else -1)
        print(fname)
        # now actually write it
        lines_written = 0
        fname = os.path.join(directory, fname)
        with codecs.open(fname, encoding='utf-8', mode='w') as f:
            for line in data_slice:
                f.write(line)
                lines_written += 1
        logging.info("wrote %d lines to '%s'", lines_written, fname)
