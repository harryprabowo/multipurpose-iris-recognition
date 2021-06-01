from app.segmentation import Segmentation
from app.upscale import SuperResolution
from app.classifier import Classifier

import cv2
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte

iris = None

def segment(img):
    segmentation = Segmentation()
    img, bbox, anded, _ = segmentation.run(img)
    
    cv_image = img_as_ubyte(img)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = cv2.rectangle(cv_image, (bbox[2], bbox[1]), (bbox[3], bbox[0]), (255, 0, 0), 5)

    # cv2.imshow("Segmentation", cv_image)
    # #waits for user to press any key
    # #(this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)

    # # #closing all open windows
    # cv2.destroyAllWindows()

    return anded, cv_image, (bbox[2], bbox[1], bbox[3], bbox[0])

def sr(sr_image):
    super_resolution = SuperResolution()
    sr_image = super_resolution.upscale(sr_image)
    return sr_image

def identify(iris):
    classifier = Classifier()
    return classifier.identify(iris)

if __name__ == '__main__':
    img = imread('.\\datasets\\C1_S1_I1.tiff')
    global_img = img

    anded, cv_image, (x,y,w,h) = segment(img)

    from numba import cuda
    device = cuda.get_current_device()
    device.reset()

    sr_image = img_as_ubyte(anded)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)

    iris = sr(sr_image)

    # cv2.imshow("SR", iris)
    # #waits for user to press any key
    # #(this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)

    # #closing all open windows
    # cv2.destroyAllWindows()

    # iris = cv2.imread(".\\datasets\\segmented.png")
    print(identify(iris))

    show_img = cv2.putText(cv_image, "C1", (x-20, y-30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("IRIS RECOGNITION", cv_image)

    # #waits for user to press any key
    # #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # #closing all open windows
    cv2.destroyAllWindows()
