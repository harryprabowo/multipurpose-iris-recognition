import os
import numpy as np
from skimage import color

import core.modules.IrisSeg.Model.utils as utils
import core.modules.IrisSeg.Model.model as modellib
from core.modules.IrisSeg.Model.config import Config


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
            IMAGES_PER_GPU = 2

            NUM_CLASSES = 1 + 1  # background + 3 shapes

            IMAGE_MIN_DIM = 64
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

        return img, bbox, anded, r