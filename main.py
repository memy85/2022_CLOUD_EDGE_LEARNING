# ----------------------------------------------  Server Script

from pathlib import Path
import os, sys

PROJECT_DIR = ''
os.sys.path.append(PROJECT_DIR)

from cloud2edge.manager import manager
from cloud2edge.communication import communicator, sender, receiver, config
from cloud2edge.aggregator import aggregator
from cloud2edge.models.server import server

# set preconditions


# ----------------------------------------------  Client Script


if __name__ == '__main__':
    communicator