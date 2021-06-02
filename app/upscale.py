import os
from re import I
import numpy as np
import torch

from basicsr.archs.edsr_arch import EDSR

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import time
from os import walk, path
from datetime import datetime

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

def calc_metric(hr, sr):
    sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]))
    psnr = peak_signal_noise_ratio(hr, sr)
    ssim = structural_similarity(hr, sr, multichannel=True)
    return psnr, ssim


def evaluate(hr, module = SuperResolution(), img_name="test"):
    scale_percent = 50  # percent of original size
    width = int(hr.shape[1] * scale_percent / 100)
    height = int(hr.shape[0] * scale_percent / 100)
    dim = (width, height)

    lr = cv2.resize(hr, dim)

    start = time.time()    
    sr = module.upscale(lr)
    end = time.time()

    cv2.imwrite('test\\upscale\\sr\\' + img_name + ".tiff", sr)

    psnr, ssim = calc_metric(hr, sr)
    
    return psnr, ssim, end-start


def main(dir):
    module = SuperResolution()

    root, _, filenames = next(walk(dir))
    avg_psnr = 0
    avg_ssim = 0
    avg_exec_time = 0
    i = 0

    with open(os.path.join("test", "upscale", "log." + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"), 'a+') as fp:
        fp.write("Image,PSNR,SSIM,Exec Time\n")

        for filename in filenames:
            img, ext = path.splitext(filename)

            if(ext not in ['.tiff', '.png', '.jpg']):
                continue

            try:
                hr = cv2.imread(os.path.join(root, filename))

                psnr, ssim, exec_time = evaluate(hr, module, img)
                print(img, psnr, ssim, exec_time)
                
                fp.write('{},{},{},{}\n'.format(img, psnr, ssim, exec_time))

                avg_psnr += psnr
                avg_ssim += ssim
                avg_exec_time += exec_time
                i += 1
            except Exception as e:
                print(e)
                continue

            # break

        avg_psnr /= i
        avg_ssim /= i
        avg_exec_time /= i

        fp.write('Average,{},{},{}\n'.format(avg_psnr, avg_ssim, avg_exec_time))
        fp.close()

if __name__ == "__main__":
    main("..\\datasets")
