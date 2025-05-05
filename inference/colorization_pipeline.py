import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import time

# Check if GPU is available and use Metal Performance Shaders on macOS
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Metal Performance Shaders
    print("Using device: Metal Performance Shaders (MPS)")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())
    

class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=256, model_size='large'):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        start_time = time.time()  # Start time

        self.height, self.width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)  # Ensure 3 channels
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.height, self.width)

        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        # Resize output_bgr to match the original image dimensions
        output_bgr_resized = cv2.resize(output_bgr, (self.width, self.height))

        # Calculate PSNR
        psnr_value = calculate_psnr(img, output_bgr_resized)
        print(f"PSNR: {psnr_value:.3f}")

        # Calculate SSIM
        ssim_value = calculate_ssim(img, output_bgr_resized)
        print(f"SSIM: {ssim_value:.3f}")

        end_time = time.time()  # End time
        print(f"Time taken for colorization: {end_time - start_time:.2f} seconds")

        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/net_g_200000.pth')
    parser.add_argument('--input_file', type=str, help='input image file path')
    parser.add_argument('--output_file', type=str, default='result.png', help='output image file path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    img = cv2.imread(args.input_file)
    if img is None:
        print(f"Error: Unable to read image file {args.input_file}")
        return

    image_out = colorizer.process(img)
    cv2.imwrite(args.output_file, image_out)
    print(f"Output saved to {args.output_file}")


if __name__ == '__main__':
    main()
