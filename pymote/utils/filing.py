__author__ = 'farrukh'

"""
Helper module to organize files in appropriate folders
"""
import os
import datetime

date2str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%dT%H-%M-%S")
path = os.path.abspath(os.path.curdir)

DATA_DIR = os.path.join(path, "data")

CHART_DIR = os.path.join(path, "charts")

TOPOLOGY_DIR = os.path.join(path, "topology")

LOG_DIR = os.path.join(path, "logs")

DATETIME_DIR = os.path.join(path, date2str)


def get_path(d, f, prefix=None):
    if not os.path.exists(d):
        os.mkdir(d)
    if prefix:
        return os.path.join(d,f+str(prefix))
    else:
        return os.path.join(d,f)

