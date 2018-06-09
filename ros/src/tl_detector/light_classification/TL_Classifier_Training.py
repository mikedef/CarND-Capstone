import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
cwd = os.getcwd()

%matplotlib inline

import io
import base64
from IPython.display import HTML

#set image color space
colorspace = cv2.COLOR_BGR2RGB

import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC, SVC
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.cross_validation import train_test_split
import time
from scipy.ndimage.measurements import label

# from moviepy.editor import VideoFileClip
from IPython.display import HTML

import glob

from skimage import data, color, exposure
from skimage.feature import hog

spatial_size = (15,32)
