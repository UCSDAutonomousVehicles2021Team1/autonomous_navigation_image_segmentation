import os
import subprocess
import glob
import numpy as np

def move_data(rawdir, storedir):
    """
    >>> posedir = "data/poses/"
    >>> os.path.isfile(posedir+'param1_gt.txt')
    True
    >>> os.path.isfile(posedir+'param2_gt.txt')
    True
    >>> os.path.isfile(posedir+'param1_slam.txt')
    True
    >>> os.path.isfile(posedir+'param1_odom.txt')
    True
    """
    #Creates the directory to copy data into
    os.makedirs(storedir, exist_ok = True)
    #For every folder in the directory provided by the user
    for folder in os.listdir(rawdir):
        #Copy the folder into our own repository
        subprocess.call(["cp", "-a", os.path.join(rawdir + folder), os.path.join(storedir)])