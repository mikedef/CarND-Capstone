import numpy as np
import cv2
import os
import time
import random
import matplotlib.image as mpimg
import scipy.misc
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skimage.feature import hog

def train(savFileName):
    data = sio.loadmat('images/featureVector.mat')
    red_features = data['red_features']
    yellow_features = data['yellow_features']
    green_features = data['green_features']
    no_features = data['no_features']

    # print(red_features.shape)
    # print(yellow_features.shape)
    # print(green_features.shape)
    # print(no_features.shape)
    X = np.vstack((red_features, yellow_features, green_features, no_features)).astype(np.float64)#.reshape(-1,1)
    # X_scaler = StandardScaler().fit(X)
    # # pickle.dump(X_scaler,'images/Scaler.pkl')
    # # Apply the scaler to X
    # scaled_X = X_scaler.transform(X)
    y = np.hstack((np.zeros(len(red_features)), np.ones(len(yellow_features)), 2*np.ones(len(green_features)), 3*np.ones(len(no_features))))

    print(X.shape)
    print(y.shape)

    rand_state = np.random.randint(0, 100)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     scaled_X, y, test_size=0.2, random_state=rand_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    print('Training data size: ', len(X_train))
    print('Test data size: ', len(X_test))

    svc = pipeline = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC(C = 0.01))])
    # svc.set_params(svc__C=0.01)
    # svc = LinearSVC(C = 0.01)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # savFileName = 'images/TrafficLightSVC.sav'
    if os.path.isfile(savFileName):
        os.remove(savFileName)
    pickle.dump(svc, open(savFileName, 'wb'))

def change_cspace(image,color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
    else:
        feature_image = np.copy(image)
    return feature_image

def bin_spatial(image, size=(32,32)):
    features = cv2.resize(image, size).ravel()
    return features

def color_hist(image, nbins=32):
    if len(image.shape) > 1:
        features = []
        for i in range(image.shape[2]):
            # Compute the histogram of the color channels separately
            hist,edges = np.histogram(image[:, :, i], bins=nbins)#, range = (0,256))
            features.append(hist)
        hist_features = np.concatenate(features)
    else:
        hist_features, edges = np.histogram(image, bins=nbins)#, range = (0,256))

    return hist_features

def get_hog_features(image_in, hog_channel = "ALL", orient = 9, pix_per_cell = 8, cell_per_block = 2, vis=False, feature_vec=True):
    #image = np.copy(image_in)
    image = cv2.resize(np.copy(image_in), spatial_size)#
    if len(image.shape) > 1 and hog_channel == "ALL":
        features = []
        vis_image = None
        for i in range(image.shape[2]):
            # Compute the histogram of the color channels separately
            hog_vals, hog_image = hog(image[:,:,i], orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),visualise=True)#, transform_sqrt=True, feature_vector=feature_vec)
            if vis_image is None:
                vis_image = hog_image
            else:
                vis_image = np.dstack((vis_image,hog_image))
            features.append(hog_vals)
        hog_features = np.concatenate(features)
    elif len(image.shape) > 1 and hog_channel != "ALL":
        hog_vals, hog_image = hog(image[:,:,hog_channel], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=True, feature_vector=feature_vec)
        hog_features = hog_vals
        vis_image = hog_image
    else:
        hog_vals, hog_image = hog(image, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=True, feature_vector=feature_vec)
        hog_features = hog_vals
        vis_image = hog_image

    if vis is True:
        return hog_features, hog_image
    else:
        return hog_features

def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=2, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    img = np.copy(image)#cv2.resize(image, spatial_size)
    feature_image = change_cspace(img,color_space)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        hog_features = get_hog_features(feature_image, hog_channel = hog_channel, orient = orient,
                                        pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,
                                        vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, color_space='RGB',
                    spatial_size=(32,32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = [[] for i in range(4)]

    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (15, 32))
        #4) Extract features for that window using single_img_features()
        ftrs = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # #5) Scale extracted features to be fed to classifier
        # test_features = scaler.transform(np.array(ftrs).reshape(1, -1))
        test_features = np.array(ftrs).reshape(1, -1)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction in [0] and (not prediction in [1,2,3]):
            on_windows[0].append(window)
        elif prediction == 1 and (not prediction in [0,2,3]):
            on_windows[1].append(window)
        elif prediction == 2 and (not prediction in [0,1,3]):
            on_windows[2].append(window)
        if prediction != 3:
            on_windows[3].append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap_thresh = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap_thresh[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap_thresh

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 25, 255), 6)
    # Return the image
    return img

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(image_shape, x_start_stop=[None, None], y_start_stop=[None, None],
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

def average_slide_windows(image_shape, x_start_stop=[None, None], y_start_stop=[None, None], xy_overlap=(0.5, 0.5)):
    windows = []
    for xy in [sliding_window_size]:#[128]:#, 96, 140]:
        window = slide_window(image_shape, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                    xy_window=(xy/2, xy), xy_overlap=xy_overlap)
        windows += window
    return windows

def classify(imgFileName):
    spl = imgFileName.split('_')
    img = mpimg.imread(imgFileName)
    r = img.shape[0]
    c = img.shape[1]
    if float(spl[-2]) >=  0 and float(spl[-2]) <= 8:
        patch_start = (0,0)
        patch_size = (c,r/2)
    elif float(spl[-2]) >  8 and float(spl[-2]) <= 20:
        patch_start = (0,r/5)
        patch_size = (c,4*r/7)
    elif float(spl[-2]) >  20 and float(spl[-2]) <= 40:
        patch_start = (0,r/3)
        patch_size = (c,5*r/12)
    elif float(spl[-2]) >  40 and float(spl[-2]) <= 65:
        patch_start = (0,r/2)#2*r/5)
        patch_size = (c,r/3)#2*r/5)
    else: #float(spl[-2]) >  65 and float(spl[-2]) <= 75:
        patch_start = (0,3*r/5)
        patch_size = (c,7*r/30)

    image = np.copy(img)

    # f,ax = plt.subplots(1)
    # rect = patches.Rectangle(patch_start,patch_size[0],patch_size[1],linewidth=1,edgecolor='r',facecolor='none')
    # ax.imshow(img)
    # ax.add_patch(rect)
    # plt.title(imgFileName.split('/')[-1])
    # plt.show()

    # # Min and max in y to search in slide_window()
    y_start_stop = [patch_start[1],patch_start[1]+patch_size[1]] ## Dist ~ [15,30]

    windows = average_slide_windows(image.shape, x_start_stop=[None, None], y_start_stop=y_start_stop,xy_overlap=xy_overlap)
    all_windows = windows
    # tf_windows = [all_windows,[]]
    tf_windows = search_windows(image, all_windows, svc, color_space=cspace,
                                spatial_size=spatial_size, hist_bins=nbins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
    #     hot_windows.append(windows)

    # f,ax = plt.subplots(1)
    # ax.imshow(image)
    # color = ['r','y','g']
    window_img = np.copy(image)
    colors = [(255,0,0),(255,255,0),(0,255,0)]
    for i in range(len(tf_windows)-1):
        # print(tf_windows[i])
        window_img = draw_boxes(window_img, tf_windows[i], color=colors[i], thick=3)
        # patch_start = tuple(tf_windows[i][0])
        # patch_size = (tf_windows[i][1][0] - tf_windows[i][0][0], tf_windows[i][1][1] - tf_windows[i][0][1])
        # rect = patches.Rectangle(patch_start,patch_size[0],patch_size[1],linewidth=1,edgecolor=color[i],facecolor='none')
        # ax.add_patch(rect)

    scipy.misc.imsave('images/Traffic_Light_Images_Annotated4/' + imgFileName.split('/')[-1], window_img)
    # plt.imshow(window_img)
    # plt.title(imgFileName.split('/')[-1])
    # plt.show()

### dist in [0,8]=> patch (0,0) -> (c,r/2)
### dist in [8,20] => patch (0,r/5) -> (c,3*r/7)
### dist in [20,40] => patch (0,r/3) -> (c,5*r/12)
### dist in [40,65] => patch (0,2*r/5) -> (c,2*r/5)
### dist in [65,75] => patch (0,3*r/5) -> (c,7*r/30)

def draw_bounding_boxes(test_bounding_boxes):
    tl_image_dir = ['images/Traffic_Light_Images_Distance']
    images = os.listdir(tl_image_dir[0])
    print("Total Number of images: ", len(images))
    for names in images:
        if names.endswith(".png") or names.endswith(".jpg"):
            spl = names.split('_')
            if test_bounding_boxes:
                if float(spl[-2]) >= 40 and float(spl[-2]) <= 65:
                    img = mpimg.imread(os.path.join(tl_image_dir[0],names))
                    r = img.shape[0]
                    c = img.shape[1]
                    patch_start = (0,2*r/5)
                    patch_size = (c,2*r/5)
                    f,ax = plt.subplots(1)
                    rect = patches.Rectangle(patch_start,patch_size[0],patch_size[1],linewidth=1,edgecolor='r',facecolor='none')
                    ax.imshow(img)
                    ax.add_patch(rect)
                    plt.title(names)
                    plt.show()
                    # time_str = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S:%f")
                    # # print(np.shape(img_1))
                    # scipy.misc.imsave('Not_Traffic_Light_Images/' + time_str + '_3.jpg', img_1)
            else:
                # if float(spl[-2]) >= 0 and float(spl[-2]) <= 8:
                # if float(spl[-2]) > 8 and float(spl[-2]) <= 20:
                # if float(spl[-2]) > 20 and float(spl[-2]) <= 40:
                # if float(spl[-2]) > 40 and float(spl[-2]) <= 65:
                if float(spl[-2]) > 65 and float(spl[-2]) <= 75:
                    classify(os.path.join(tl_image_dir[0],names))


test_bounding_boxes = False

### dist in [0,8]=> sliding_window_size = 220, xy_overlap = (0.9,0.7)
### dist in [8,20] => sliding_window_size = 150, xy_overlap = (0.7,0.7)
### dist in [20,40] => sliding_window_size = 120, xy_overlap = (0.7,0.5)
### dist in [40,65] => sliding_window_size = 90, xy_overlap = (0.5,0.5)
### dist in [65,75] => sliding_window_size = 70, xy_overlap = (0.5,0.5)

sliding_window_size = 70
xy_overlap = (0.5,0.5)#(0.7,0.5)
spatial_size = (32,15)
cspace = 'RGB'#'HSV'#'YCrCb'#
nbins = 32
orient = 9
pix_per_cell = 4
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
vis = False
feature_vec = True
spatial_feat = True
hist_feat = True
hog_feat = True

savFileName = 'images/TrafficLightSVC_1.sav'
if not os.path.isfile(savFileName):
    train(savFileName)
# load the model from disk
svc = pickle.load(open(savFileName, 'rb'))
draw_bounding_boxes(test_bounding_boxes)
