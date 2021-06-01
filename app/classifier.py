import cv2
import keras
import numpy as np
import os
import pickle

IMAGE_SIZE = (200, 200)

class Classifier:
    def __init__(self, root_dir=os.getcwd()):
        self.ROOT_DIR = root_dir

        print('ooooooooooooooooooooooooo1', root_dir +
              "\\core\\modules\\Classifier\\ubiris_deepnet201.h5")

        self.model = keras.models.load_model(
            root_dir + "\\core\\modules\\Classifier\\ubiris_deepnet201.h5")
        print('ooooooooooooooooooooooooo2')
        self.bottleneck_model = keras.models.load_model(
            root_dir + "\\core\\modules\\Classifier\\bottleneck_model.h5")
        print('ooooooooooooooooooooooooo3')

        self.load_le_map()
        print('ooooooooooooooooooooooooo4')
    
    def load_le_map(self):
        pickle_in = open(
            self.ROOT_DIR + "\\core\\modules\\Classifier\\le_map.pickle", "rb")
        self.le_map = pickle.load(pickle_in)

    def identify(self, img):
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.array(img)/255
        img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        img = self.bottleneck_model.predict(img)
        img = np.array(img)

        identify = self.model.predict_classes([img, ])

        return self.le_map[identify[0]]

if __name__ == "__main__":
    classifier = Classifier(os.path.join(os.getcwd(), '..'))
    img = cv2.imread("..\\core\\modules\\IrisSeg\\segmented\\C2_S1_I7.png")

    print(classifier.identify(img))
