import os
import numpy as np
import torch

from basicsr.archs.edsr_arch import EDSR


class SuperResolution:
    def __init__(self, model_path='\\core\\modules\\BasicSR\\experiments\\pretrained_models\\EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth'):
        # Root directory of the project
        ROOT_DIR = os.getcwd()

        self.device = torch.device('cuda')
        self.model_path = ROOT_DIR + model_path
        self.init_model()

    def init_model(self):
        # set up model
        model = EDSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=256,
            num_block=32,
            upscale=2,
            res_scale=0.1,
            img_range=255.0,
            rgb_mean=[0.4488, 0.4371, 0.404]
        )

        model.load_state_dict(torch.load(self.model_path)
                              ['params'], strict=True)
        model.eval()

        self.model = model.to(self.device)

    def upscale(self, image):
        img = image.astype(np.float32) / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        # inference
        with torch.no_grad():
            output = self.model(img)

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        return output