import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.edsr_arch import EDSR

# configuration
model_path = '.\\..\\experiments\\pretrained_models\\EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth'
folder = '.\\..\\datasets'
device = 'cuda'

device = torch.device(device)

# set up model
model = EDSR(
    num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=2, res_scale=0.1, img_range=255.0, rgb_mean=[0.4488, 0.4371, 0.404])
model.load_state_dict(torch.load(model_path)['params'], strict=True)
model.eval()
model = model.to(device)

os.makedirs('results/EDSR', exist_ok=True)
for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
    imgname = os.path.splitext(os.path.basename(path))[0]
    print(idx, imgname)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                        (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        output = model(img)
    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'results/EDSR/{imgname}_EDSR.png', output)
    break
