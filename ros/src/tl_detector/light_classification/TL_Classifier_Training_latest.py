import os
from TLClassifier import TLClassifier as TLclf
import scipy.io as sio
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import scipy

TLC = TLclf()
TLC.useCanny = True
# TLC.useCanny = False

if TLC.useCanny:
    outFile = 'images/featureVector_withCanny.mat'
else:
    outFile = 'images/featureVector_withoutCanny.mat'

if not os.path.isfile(outFile):
    tl_image_dir = ['traffic_light_images/training/red',
                    'traffic_light_images/training/yellow',
                    'traffic_light_images/training/green',
                    'images/Not_Traffic_Light_Images',
                    'traffic_light_images/red',
                    'traffic_light_images/yellow',
                    'traffic_light_images/green',
                    'traffic_light_images/Unknown']

    labels = [0,1,2,3]
    label_txt = ['Red','Yellow','Green','No']

    images = [[] for i in labels]

    for i in range(len(tl_image_dir)):
        j = i%4
        image_names = os.listdir(tl_image_dir[i])
        for names in image_names:
            if names.endswith(".png") or names.endswith(".jpg"):
                img = mpimg.imread(os.path.join(tl_image_dir[i],names))
                images[j].append(img)

    for i in range(len(images)):
        print("Statistics for " + label_txt[i] + " light images:")
        print("# images = " + str(len(images[i])))
        print("#########################################")

    if True:#not loadMat:
        features = [[] for i in range(len(images))]

        for i in range(len(images)):
            for j in range(len(images[i])):
                ftrs = TLC.getFeatureVector(images[i][j])
                if j == 0:
                    features[i] = np.array(ftrs)
                else:
                    features[i] = np.vstack((features[i],ftrs))

        sio.savemat(outFile, {'red_features':features[0],
                                          'yellow_features':features[1],
                                          'green_features':features[2],
                                          'no_features':features[3]})
        red_features = features[0]
        yellow_features = features[1]
        green_features = features[2]
        no_features = features[3]
else:
    data = sio.loadmat(outFile)
    red_features = data['red_features']
    yellow_features = data['yellow_features']
    green_features = data['green_features']
    no_features = data['no_features']
    print(red_features.shape)
    print(yellow_features.shape)
    print(green_features.shape)
    print(no_features.shape)

X = np.vstack((red_features, yellow_features, green_features, no_features)).astype(np.float64)
y = np.hstack((np.zeros(len(red_features)), np.ones(len(yellow_features)), 2*np.ones(len(green_features)), 3*np.ones(len(no_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

if TLC.useCanny:
    pathToModelFile = 'images/TrafficLightSVC_canny.sav'
else:
    pathToModelFile = 'images/TrafficLightSVC_withoutCanny.sav'

if not os.path.isfile(pathToModelFile):
    TLC.trainClassifier(X_train, X_test, y_train, y_test, pathToModelFile)
TLC.setCLFModel(pathToModelFile)

tl_image_dir = ['images/Traffic_Light_Images_Distance']
images = os.listdir(tl_image_dir[0])
print("Total Number of images: ", len(images))
for names in images:
    if names.endswith(".png") or names.endswith(".jpg"):
        spl = names.split('_')
        if True:#float(spl[-2]) > 0 and float(spl[-2]) <= 8:
        # if float(spl[-2]) > 8 and float(spl[-2]) <= 20:
        # if float(spl[-2]) > 20 and float(spl[-2]) <= 40:
        # if float(spl[-2]) > 40 and float(spl[-2]) <= 65:
        # if float(spl[-2]) > 65:# and float(spl[-2]) <= 40:
            img = mpimg.imread(os.path.join(tl_image_dir[0],names))
            delta_wp = float(spl[-2])
            all_windows = TLC.get_windows(img,delta_wp)
            #tf_windows = [all_windows,[]]
            tf_windows = TLC.search_windows(img, all_windows)
            window_img = np.copy(img)
            colors = [(255,0,0),(255,255,0),(0,255,0)]
            maxLen = max(len(p) for p in tf_windows)
            if maxLen == 0:
                continue
            for i in range(len(tf_windows)):
                if len(tf_windows[i]) == maxLen:
                    # print(tf_windows[i])
                    window_img = TLC.draw_boxes(window_img, tf_windows[i], color=colors[i], thick=3)
                    break

            scipy.misc.imsave('images/Traffic_Light_Images_Annotated4/' + names.split('/')[-1], window_img)
            # ## print(names, TLC.predict(img))
            # f,ax = plt.subplots(1)
            # ax.imshow(window_img)
            # plt.title(names.split('/')[-1])
            # plt.show()
