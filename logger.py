from pymote.utils.filing import get_path, LOG_DIR
import os

LOG_CONFIG = {
              'version': 1,
              'loggers':
              {
                 'pymote':
                 {
                    'level': 'DEBUG',
                    'handlers': ['fileHandler', 'consoleHandler']
                 },
                 'pymote.simulation':
                 {
                    'level': 'DEBUG',
                    'handlers': ['simFileHandler'],
                    'propagate': 1
                 }
               },
               'handlers':
               {
                    'fileHandler':
                    {
                        'class': 'logging.FileHandler',
                        'level': 'DEBUG',
                        'formatter': 'fileFormatter',
                        'filename': get_path(LOG_DIR, 'pymote.log')
                    },
                    'consoleHandler':
                    {
                        'class': 'logging.StreamHandler',
                        'formatter': 'consoleFormatter',
                        'stream': 'ext://sys.stdout'
                    },
                    'simFileHandler':
                    {
                        'class': 'logging.FileHandler',
                        'level': 'DEBUG',
                        'formatter': 'fileFormatter',
                        'filename': get_path(LOG_DIR, 'simulation.log')
                    }
               },
               'formatters':
               {
                    'fileFormatter':
                    {
                        'format': ('%(asctime)s - %(levelname)s:'
                                   '[%(filename)s] %(message)s'),
                        'datefmt': '%Y-%m-%d %H:%M:%S'
                    },
                    'consoleFormatter':
                    {
                        'format': ('%(asctime)s - %(levelname)-8s'
                                   '[%(filename)s: %(lineno)s]: %(message)s'),
                        #'datefmt': '%Y-%m-%d %H:%M:%S'
                    }
                }
              }
import logging.config
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger('pymote')
