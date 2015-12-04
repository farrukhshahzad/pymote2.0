__author__ = 'farrukh'

"""
Helper module to organize files in appropriate folders
"""
import os
import datetime
import ConfigParser

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


def load_config(ini_file):
    config = ConfigParser.ConfigParser()
    read_files = config.read(ini_file)
    if len(read_files) == 0:
        raise Exception("Failed to read %s" % ini_file)
    meta = {}
    for item in config.options("meta"):
        meta[item] = config.get("meta", item)
    return meta


def load_metadata():
    # default used in case can't read METADATA
    meta = {"name": "pymote", "version": "2.0.x"}
    try:
        path = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(path, '../..', 'METADATA')
        meta.update(load_config(ini_file))
    except:
        pass
    return meta

def getDateStr(date):
    return datetime.datetime.strftime(date, "%Y-%m-%dT%H-%M-%S")