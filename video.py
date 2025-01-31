import sys
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
import numpy as np
from tqdm import tqdm

# Load TensorFlow Hub model for Neural Style Transfer
style_transfer_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Function to preprocess an image for NST model
def preprocess_image(image, target_dim=(256, 256)):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_dim)
    image = image[tf.newaxis, :]
    return image

# Function to apply style transfer to a video
def apply_style_to_video(video_path, style_image_path, output_path, num_frames=None):
    # Load and preprocess style image
    style_image = tf.io.read_file(style_image_path)
    style_image = tf.image.decode_image(style_image, channels=3)
    style_image = preprocess_image(style_image)

    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_to_process = min(num_frames, total_frames) if num_frames else total_frames

    # Initialize video writer
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {frames_to_process}/{total_frames} frames...")

    try:
        for frame_idx in tqdm(range(frames_to_process)):
            ret, frame = video.read()
            if not ret:
                break

            # Convert frame to tensor
            content_image = tf.convert_to_tensor(frame, dtype=tf.float32)
            content_image = preprocess_image(content_image)

            # Apply style transfer
            stylized_image = style_transfer_model(content_image, style_image)[0]

            # Convert back to NumPy
            stylized_image = tf.image.resize(stylized_image, (height, width))
            stylized_frame = tf.cast(stylized_image * 255, tf.uint8).numpy()[0]

            # Save to video
            out.write(cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR))

    finally:
        video.release()
        out.release()
        print(f"Stylized video saved to: {output_path}")

# Main function
if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    style_image_path = input("Enter the path to the style image: ")
    output_path = input("Enter the path for the output video (default: output.mp4): ") or "output.mp4"
    num_frames = input("Enter number of frames to process (optional, press Enter for all frames): ")
    num_frames = int(num_frames) if num_frames else None

    apply_style_to_video(video_path, style_image_path, output_path, num_frames)