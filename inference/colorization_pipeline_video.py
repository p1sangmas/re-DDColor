import cv2
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import time

# ========================================
# Helper Functions and Colorization Pipeline
# ========================================

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

class ImageColorizationPipeline:
    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.height, self.width = img.shape[:2]
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()

        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        return output_img


# ========================================
# Process Workflow
# ========================================

def extract_frames(video_path, output_dir):
    """Extracts frames from a video and saves them as `.jpg` files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    cap.release()
    print(f"Extracted {frame_number} frames to {output_dir}.")


def colorize_directory(input_dir, output_dir, model_path, input_size, model_size):
    """Colorizes all `.jpg` images in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colorizer = ImageColorizationPipeline(model_path=model_path, input_size=input_size, model_size=model_size)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    for image_file in tqdm(image_files, desc="Colorizing frames"):
        input_path = os.path.join(input_dir, image_file)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Unable to read image {input_path}.")
            continue

        colorized_img = colorizer.process(img)
        output_path = os.path.join(output_dir, f"colorized_{image_file}")
        cv2.imwrite(output_path, colorized_img)

    print(f"Colorized frames saved to {output_dir}.")


def combine_frames_to_video(frame_dir, output_video_path, fps=30):
    """Combines `.jpg` frames into a `.mp4` video."""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print(f"No `.jpg` frames found in {frame_dir}.")
        return

    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Unable to read frame {first_frame_path}.")
        return

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Combining frames"):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Unable to read frame {frame_path}.")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}.")


# ========================================
# Main Execution
# ========================================

def main(args):
    # Step 1: Extract frames from the video
    extract_frames(args.video_path, args.frame_dir)

    # Step 2: Colorize the extracted frames
    colorize_directory(args.frame_dir, args.colorized_dir, args.model_path, args.input_size, args.model_size)

    # Step 3: Combine colorized frames into a video
    combine_frames_to_video(args.colorized_dir, args.output_video_path, args.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video: extract, colorize, and combine frames.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--frame_dir', type=str, default="extracted_frames", help="Directory to save extracted frames.")
    parser.add_argument('--colorized_dir', type=str, default="colorized_frames", help="Directory to save colorized frames.")
    parser.add_argument('--output_video_path', type=str, default="output_video.mp4", help="Path to save the output video.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the DDColor model weights.")
    parser.add_argument('--input_size', type=int, default=512, help="Input size for the DDColor model.")
    parser.add_argument('--model_size', type=str, default='large', help="DDColor model size (tiny or large).")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second for the output video.")
    args = parser.parse_args()

    main(args)
