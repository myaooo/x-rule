"""
Configurations of the module
"""

import os

# Constants

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../'))
ENV = 'production'


def root_dir():
    return ROOT_DIR


def mode(new_mode=None):
    global ENV
    if new_mode is None:
        return ENV
    elif new_mode in ['production', 'development']:
        ENV = new_mode
