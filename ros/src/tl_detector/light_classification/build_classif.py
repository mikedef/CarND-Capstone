import cv2

import numpy as np

import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import os
import glob


def load_dataset(image_dir):
    im_list = []
    image_types = ["red", "yellow", "green"]

    for im_type in image_types:
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            im = mpimg.imread(file)
            if im is not None:
                im_list.append((im, im_type))

    return im_list


def save_image(name, img):
    cv2.imwrite("./light_classification/test/t3" + str(name) + ".png", img)
    cv2.waitKey(0)


def standardize_input(image):
    image = image[150:450, 0:800, :]
    # print(np.array(image).shape)
    image = cv2.resize(image, (60, 160))
    # print(np.array(image).shape)
    return image


def one_hot_encode(label):
    if label == "red":
        return [1.0, 0.0, 0.0]
    elif label == "yellow":
        return [0.0, 1.0, 0.0]
    elif label == "green":
        return [0.0, 0.0, 1.0]
    return []


def standardize(image_list):
    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        standardized_im = standardize_input(image)
        one_hot_label = one_hot_encode(label)
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


def create_feature(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # Try to concatenate hsv1 and 2
    # feature = np.concatenate([rgb_image[:, :, 0].flatten(), rgb_image[:, :, 1].flatten()]) # + hsv[:,:,2]
    return color_hist(rgb_image)


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def estimate_label(rgb_image, scaler, classifier):
    ft = create_feature(rgb_image)
    ft = scaler.transform([ft])
    return classifier.predict(ft)[0]


# Image data directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(PROJECT_DIR, "traffic_light_simu")
IMAGE_DIR_TRAINING = os.path.join(IMAGE_DIR, "training")
IMAGE_DIR_TEST = os.path.join(IMAGE_DIR, "test")

IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)
STANDARDIZED_LIST = standardize(IMAGE_LIST)
STANDARDIZED_TEST_LIST = standardize(load_dataset(IMAGE_DIR_TEST))

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

X = [create_feature(i[0]) for i in STANDARDIZED_LIST]

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

Y = [i[1] for i in STANDARDIZED_LIST]

clf = RandomForestClassifier(random_state=0, max_depth=18)
# clf = MLPClassifier(random_state=0, hidden_layer_sizes=150)
clf.fit(scaled_X, Y)

joblib.dump(clf, 'randomForest_trafficLight.pkl')
joblib.dump(X_scaler, 'scaler.pkl')


def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im, X_scaler, clf)
        assert (len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels

        # print(predicted_label)
        # print(true_label)
        # print("predicted_label = {} _ true_label = {}".format(predicted_label, true_label))
        if predicted_label[0] != true_label[0] or \
                predicted_label[1] != true_label[1] or \
                predicted_label[2] != true_label[2]:
            # print(predicted_label)
            # print(true_label)
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
        # else:
        #     print(
        #         "AAAAAAAAAAAAAAAAAAAAAAAAAA predicted_label = {} _ true_label = {}".format(predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct / total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))
