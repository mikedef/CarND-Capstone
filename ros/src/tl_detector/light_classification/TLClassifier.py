import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
cwd = os.getcwd()

import io
import base64
from IPython.display import HTML
import scipy
#set image color space
colorspace = cv2.COLOR_BGR2RGB

import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.cross_validation import train_test_split
import time
from scipy.ndimage.measurements import label

# from moviepy.editor import VideoFileClip
from IPython.display import HTML

import glob
import datetime
from skimage import data, color, exposure
from skimage.feature import hog

class TLClassifier:
    def __init__(self):
        self.labels = [0,1,2,3]
        self.spatial_size = (15,32)#(32,15)#32)
        self.cspace = 'RGB'#'HSV'#'YCrCb'#
        self.nbins = 32
        self.orient = 9
        self.pix_per_cell = 4
        self.cell_per_block = 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        self.feature_vec = True
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True
        self.useCanny = True
        self.cannyLowThresh = 50
        self.cannyUppThresh = 70
        self.CLFModel = None
        self.data_collection = False

    def setCLFModel(self, pathToModelFile):
        self.CLFModel = pickle.load(open(pathToModelFile, 'rb'))

    def getSlidingWindowSettings(self,delta_wp):
        if delta_wp in range(0,9):
            sliding_window_size = 220
            xy_overlap = (0.9,0.7)
        elif delta_wp in range(9,21):
            sliding_window_size = 150
            xy_overlap = (0.7,0.7)
        elif delta_wp in range(21,41):
            sliding_window_size = 120
            xy_overlap = (0.7,0.5)
        elif delta_wp in range(41,66):
            sliding_window_size = 90
            xy_overlap = (0.5,0.5)
        else: #delta_wp in [65,75]:
            sliding_window_size = 70
            xy_overlap = (0.5,0.5)
        return sliding_window_size, xy_overlap

    def getImageMask(self,delta_wp,r,c):
        if delta_wp in range(0,9):
            patch_start = (0,0)
            patch_size = (c,r/2)
        elif delta_wp in range(9,21):
            patch_start = (0,r/5)
            patch_size = (c,4*r/7)
        elif delta_wp in range(21,41):
            patch_start = (0,r/3)
            patch_size = (c,5*r/12)
        elif delta_wp in range(41,66):
            patch_start = (0,r/2)#2*r/5)
            patch_size = (c,r/3)#2*r/5)
        else: #delta_wp >  65 and delta_wp <= 75:
            patch_start = (0,3*r/5)
            patch_size = (c,7*r/30)

        # # Min and max in y to search in slide_window()
        y_start_stop = [patch_start[1],patch_start[1]+patch_size[1]]
        return y_start_stop

    def trainClassifier(self,X_train, X_test, y_train, y_test, pathToModelFile):
        clf = pipeline = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC(C = 0.01))])
        # clf = RandomForestClassifier(n_estimators=250, random_state=0)
        t=time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train clf...')

        # Check the score of the clf
        print('Test Accuracy of clf = ', round(clf.score(X_test, y_test), 4))

        if os.path.isfile(pathToModelFile):
            os.remove(pathToModelFile)
        pickle.dump(clf, open(pathToModelFile, 'wb'))

    def search_windows(self, img, windows):
        #1) Create an empty list to receive positive detection windows
        on_windows = [[] for i in range(len(self.labels)-1)]

        #2) Iterate over all windows in the list
        #for window in windows:
        for i in range(len(windows)):
            window = windows[i]
            win_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            prediction = self.predict(win_img)
            if self.data_collection:
                time_str = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S:%f")
                scipy.misc.imsave('images/Traffic_Light_Images_Annotated4/'+time_str + str(prediction[0])+'.jpg', win_img)
            #7) If positive (prediction == 1) then save the window
            if prediction in [0] and (not prediction in [1,2,3]):
                on_windows[0].append(window)
            elif prediction == 1 and (not prediction in [0,2,3]):
                on_windows[1].append(window)
            elif prediction == 2 and (not prediction in [0,1,3]):
                on_windows[2].append(window)
            # if prediction != 3:
            #     on_windows[3].append(window)
        #8) Return windows for positive detections
        return on_windows

    def slide_window(self,image_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(32, 32), xy_overlap=(0.8, 0.8)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = image_shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = image_shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def get_windows(self, img, delta_wp):
        image_shape = img.shape
        y_start_stop = self.getImageMask(delta_wp, image_shape[0], image_shape[1])
        sliding_window_size, xy_overlap = self.getSlidingWindowSettings(delta_wp)
        windows = []
        for xy in [sliding_window_size]:#[128]:#, 96, 140]:
            window = self.slide_window(image_shape, x_start_stop=[None, None], y_start_stop=y_start_stop,
                        xy_window=(xy/2, xy), xy_overlap=xy_overlap)
            windows += window
        return windows

    def predict(self, image):
        featureVector = self.getFeatureVector(image)
        return self.CLFModel.predict(featureVector)

    def draw_boxes(self,img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def getFeatureVector(self,image):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        img = np.copy(image)#cv2.resize(image, spatial_size)

        feature_image = self.change_cspace(img)
        canny_image = self.canny(feature_image,self.cannyLowThresh,self.cannyUppThresh)
        # 3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            if self.useCanny:
                spatial_features = self.bin_spatial(canny_image)
            else:
                spatial_features = self.bin_spatial(feature_image)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = self.color_hist(feature_image)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.useCanny:
                hog_features = self.get_hog_features(feature_image)
            else:
                hog_features = self.get_hog_features(feature_image)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        ftr_vector = np.concatenate(img_features)
        return ftr_vector.reshape((1,len(ftr_vector)))

    def get_hog_features(self,image_in):
        # image = np.copy(image_in)
        image = cv2.resize(np.copy(image_in), self.spatial_size)#
        if len(image.shape) > 2 and self.hog_channel == "ALL":
            features = []
            for i in range(image.shape[2]):
                # Compute the histogram of the color channels separately
                hog_vals = hog(image[:,:,i], orientations=self.orient,
                                          pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                          cells_per_block=(self.cell_per_block, self.cell_per_block))#,visualise=True
                features.append(hog_vals)
            hog_features = np.concatenate(features)
        elif len(image.shape) > 2 and self.hog_channel != "ALL":
            hog_vals = hog(image[:,:,self.hog_channel], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block))#,visualise=True
            hog_features = hog_vals
        else:
            hog_vals = hog(image, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block))#,visualise=True
            hog_features = hog_vals

        return hog_features

    def canny(self,img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, low_threshold, high_threshold)/255.0

    def change_cspace(self,image):
        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)
        else:
            feature_image = np.copy(image)
        return feature_image

    def bin_spatial(self,image):
        features = cv2.resize(image, self.spatial_size).ravel()
        return features

    def color_hist(self,image):
        if len(image.shape) > 1:
            features = []
            for i in range(image.shape[2]):
                # Compute the histogram of the color channels separately
                hist,edges = np.histogram(image[:, :, i], bins=self.nbins)#, range = (0,256))
                features.append(hist)
            hist_features = np.concatenate(features)
        else:
            hist_features, edges = np.histogram(image, bins=self.nbins)#, range = (0,256))

        return hist_features
