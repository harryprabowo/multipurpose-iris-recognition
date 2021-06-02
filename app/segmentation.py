import os
import numpy as np
from skimage import color
from skimage.io import imread, imsave, imshow

import core.modules.IrisSeg.Model.utils as utils
import core.modules.IrisSeg.Model.model as modellib
from core.modules.IrisSeg.Model.config import Config

from os import walk, path
from datetime import datetime
import time
import cv2

GROUND_TRUTH_DIR = os.path.join('core', 'modules', 'IrisSeg', 'IRISSEG-EP-Masks')
class Segmentation:
    def __init__(self, weight_path="core\\modules\\IrisSeg\\Weights\\mask_rcnn_irises_Ubiris.h5"):
        # Root directory of the project
        ROOT_DIR = os.getcwd()
        os.chdir(ROOT_DIR)

        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        self.MODEL_PATH = os.path.join(ROOT_DIR, weight_path)

        self.prep()

    def prep(self):
        class IrisConfig(Config):
            # Give the configuration a recognizable name
            NAME = "irises"

            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            NUM_CLASSES = 1 + 1  # background + 3 shapes

            IMAGE_MIN_DIM = 640
            IMAGE_MAX_DIM = 640

            # Use a small epoch since the data is simple
            STEPS_PER_EPOCH = 1000

            VALIDATION_STEPS = 50

        class InferenceConfig(IrisConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.inference_config = InferenceConfig()
        # inference_config.display()

        self.model = modellib.MaskRCNN(
            mode="inference",
            config=self.inference_config,
            model_dir=self.MODEL_DIR
        )

        # Loads weights from a static model file
        self.model.load_weights(self.MODEL_PATH, by_name=True)

    def resize(self, image):
        im, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.inference_config.IMAGE_MIN_DIM,
            min_scale=self.inference_config.IMAGE_MIN_SCALE,
            max_dim=self.inference_config.IMAGE_MAX_DIM,
            mode="pad64"
        )

        return im

    def infer(self, image):
        r = self.model.detect([image], verbose=0)
        return r[0]

    def segment(self, image, bbox, mask):
        PADDING = 20

        cropped_mask = mask[bbox[0]-PADDING:bbox[2] +
                            PADDING, bbox[1]-PADDING:bbox[3]+PADDING]
        cropped_im = image[bbox[0]-PADDING:bbox[2] +
                           PADDING, bbox[1]-PADDING:bbox[3]+PADDING]

        cropped_mask = cropped_mask*255
        cropped_mask = color.gray2rgb(cropped_mask)

        anded = np.bitwise_and(cropped_im, cropped_mask)

        return anded, cropped_mask

    def run(self, image):
        img = self.resize(image)
        r = self.infer(img)
        bbox = r['rois'][0]
        mask = r['masks'][:, :, 0]
        anded, _ = self.segment(img, bbox, mask)

        return img, bbox, anded, r, mask


def load_ground_truth(loc):
    ground_truth = {}

    root, _, filenames = next(walk(loc))

    for filename in filenames:
        img, ext = path.splitext(filename)

        if(ext not in ['.tiff', '.png', '.jpg']):
            continue
        
        img_name = img.split("_", 1)[1]

        ground_truth[img_name] = imread(os.path.join(root, filename))

    print("Ground truth loaded successfully")

    return ground_truth


def p2f(x):
    return float(x.strip('%'))/100

def evaluate(mask_true, iris, segmentation, img_name, img_dir):
    start = time.time()    
    _, _, anded, _, mask = segmentation.run(iris)
    end = time.time()

    mask = mask*255
    mask = color.gray2rgb(mask)
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (mask_true.shape[1], mask_true.shape[0]))

    anded = anded.astype(np.uint8)

    cv2.imwrite('test\\segmentation\\masks\\' + img_name + ".tiff", mask)
    cv2.imwrite('test\\upscale\\hr\\' + img_name + ".tiff", anded)

    output = os.popen(
        os.path.join(GROUND_TRUTH_DIR, "software", "bin", "maskcmpprf.exe") +
        " -i test\\segmentation\\masks\\" + img_name + ".tiff" +
        " -i " + os.path.join(GROUND_TRUTH_DIR, "masks", "ubiris", "OperatorA_" + img_name + ".tiff")
    ).read()
    
    output = output.split()

    return p2f(output[3]), p2f(output[5]), p2f(output[7]), end-start

def main(dataset, ground_truth=GROUND_TRUTH_DIR):
    segmentation = Segmentation()
    ground_truth = load_ground_truth(os.path.join(ground_truth, "masks", "ubiris"))

    root, _, filenames = next(walk(dataset))
    avg_recall = 0
    avg_prec = 0
    avg_f1 = 0
    avg_exec_time = 0
    i = 0

    with open(os.path.join("test", "segmentation", "log." + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"), 'a+') as fp:
        fp.write("Image,Recall,Precision,F1 Measure,Exec Time\n")

        for filename in filenames:
            img, ext = path.splitext(filename)

            if(ext not in ['.tiff', '.png', '.jpg']):
                continue

            try:
                mask_true = ground_truth[img]
                iris = imread(os.path.join(root, filename))

                recall, prec, f1, exec_time = evaluate(mask_true, iris, segmentation, img, root + filename)
                print(img, recall, prec, f1, exec_time)

                fp.write('{},{},{},{},{}\n'.format(
                    img, recall, prec, f1, exec_time))

                avg_recall += recall
                avg_prec += prec
                avg_f1 += f1
                avg_exec_time += exec_time
                i += 1
            except Exception as e: 
                print(e)
                continue

            # break

        avg_recall /= i
        avg_prec /= i
        avg_f1 /= i
        avg_exec_time /= i

        fp.write('Average,{},{},{},{}\n'.format(avg_recall, avg_prec, avg_f1, avg_exec_time))

        fp.close()

if __name__ == "__main__":
    main("..\\datasets")
