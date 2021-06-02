import os
import numpy as np
import keras
import cv2
import pickle

from os import walk, path
from datetime import datetime
import time

from numpy.lib.function_base import average


IMAGE_SIZE = (200, 200)

class Classifier:
    def __init__(self, root_dir=os.getcwd()):
        self.ROOT_DIR = root_dir

        self.model = keras.models.load_model(
            root_dir + "\\core\\modules\\Classifier\\ubiris_deepnet201.h5")

        self.bottleneck_model = keras.models.load_model(
            root_dir + "\\core\\modules\\Classifier\\bottleneck_model.h5")

        self.load_le_map()
    
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

def evaluate(iris, classifier):
    start = time.time()
    pred = classifier.identify(iris)
    end = time.time()

    return pred, end-start


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(y_true, y_pred)
    prec, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return acc, prec, recall, fbeta, support

def main(dataset):
    classifier = Classifier()
    labels = set()

    root, _, filenames = next(walk("..\\UBIRIS_200_150\\CLASSES_400_300_Part1"))

    for filename in filenames:
        img, ext = path.splitext(filename)

        if(ext not in ['.tiff', '.png', '.jpg']):
            continue

        y = img.split("_")[0]
        labels.add(y)

    labels = list(labels)

    exec_time = []
    y_pred = []
    y_true = []

    root, _, filenames = next(walk(dataset))

    with open(os.path.join("test", "classifier", "log." + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"), 'a+') as fp:
        fp.write("y_true,y_pred,Exec Time\n")
        
        for filename in filenames:
            img, ext = path.splitext(filename)
            true = img.split("_")[0]

            if(ext not in ['.tiff', '.png', '.jpg']):
                continue

            try:
                iris = cv2.imread(os.path.join(root, filename))
                pred, duration = evaluate(iris, classifier)
                print(true, pred, duration)

                fp.write('{},{},{}\n'.format(
                    true, pred, duration))
                
                y_true.append(true)
                y_pred.append(pred)
                exec_time.append(duration)

            except Exception as e:
                print(e)
                continue

        fp.close()

        acc, prec, recall, fbeta, support = metrics(y_true, y_pred)
        print(  
            "Accuracy: " + str(acc) + "\n",
            "Precision: " + str(prec) + "\n",
            "Recall: " + str(recall) + ":\n",
            "FBeta Score: " + str(fbeta) + "\n",
            "Support: " + str(support)
        )

if __name__ == "__main__":
    main("..\\datasets")
