import os
from .declaration import ROOT_DIR

ROOT_DIR = ROOT_DIR()

def join_path(relative_dir):
    return os.path.normpath(os.path.join(ROOT_DIR, relative_dir))