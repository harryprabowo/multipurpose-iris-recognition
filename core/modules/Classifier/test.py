import pickle
import keras
import cv2
import numpy as np

model = keras.models.load_model("./ubiris_deepnet201.h5")
bottleneck_model = keras.models.load_model("./bottleneck_model.h5")

img_size = (200, 200)

img = cv2.imread("..\\IrisSeg\\segmented\\C2_S1_I7.png")
img = cv2.resize(img, img_size)

img = np.array(img)/255

img = img.reshape(1, img_size[0], img_size[1], 3)

img = bottleneck_model.predict(img)
img = np.array(img)

identify = model.predict_classes([img, ])

pickle_in = open("le_map.pickle", "rb")
le_map = pickle.load(pickle_in)

print(le_map[identify[0]])
