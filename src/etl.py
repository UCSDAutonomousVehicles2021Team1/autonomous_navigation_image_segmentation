import os
import subprocess

def move_data(rawdir, storedir, imagedir_name):
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
        subprocess.call(["cp", "-a", os.path.join(rawdir + folder), \
            os.path.join(storedir)])
        #Create an image directory for each folder
        os.makedirs(os.path.join(storedir, folder, imagedir_name), \
            exist_ok = True)
        #Move the image files into the image directory
        for file in os.listdir(os.path.join(storedir, folder)):
            if file != imagedir_name.strip("/") and \
            (not file.endswith('.json')):
                subprocess.call(["mv", os.path.join(storedir, folder, file), \
                    os.path.join(storedir, folder, imagedir_name, file)])
