from styx_msgs.msg import TrafficLight
from sklearn.externals import joblib
import light_classification.build_classif as builder


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.clf = joblib.load('randomForest_trafficLight.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.counter = 0

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        self.counter += 1
        # To get training images
        #builder.save_image(self.counter, image)
        standardized = builder.standardize_input(image)
        label = builder.estimate_label(standardized, self.scaler, self.clf)
        #TODO implement light color prediction

        if label[0] > 0:
            return TrafficLight.RED
        elif label[1] > 0:
            return TrafficLight.YELLOW
        elif label[2] > 0:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
