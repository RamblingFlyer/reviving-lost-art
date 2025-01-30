import sys
print(sys.executable)
import tensorflow as tf
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def apply_style_to_video(video_path, style_image_path, output_path, model_path, num_frames=None):
    """
    Applies style transfer to a specified number of frames of a video.

    Parameters:
    - video_path: Path to the input video.
    - style_image_path: Path to the style image.
    - output_path: Path to save the stylized video.
    - model_path: Path to the saved model.
    - num_frames: Number of frames to process (None for all frames).
    """
    # Load the pre-trained style transfer model
    model = tf.saved_model.load(model_path)

    # Load and preprocess the style image
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (256, 256))
        img = img[tf.newaxis, :]
        return img

    style_image = load_image(style_image_path)

    # Open the video
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Limit the number of frames to process if num_frames is specified
    frames_to_process = min(num_frames, total_frames) if num_frames else total_frames

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {frames_to_process}/{total_frames} frames...")

    for frame_idx in tqdm(range(frames_to_process)):
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess the content frame
        content_image = tf.convert_to_tensor(frame, dtype=tf.float32)
        content_image = tf.image.resize(content_image, (256, 256))
        content_image = content_image[tf.newaxis, :]

        # Apply style transfer
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

        # Postprocess and save the frame
        stylized_image = tf.image.resize(stylized_image, (height, width))
        stylized_frame = tf.cast(stylized_image * 255, tf.uint8).numpy()
        out.write(cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR))

    # Release resources
    video.release()
    out.release()
    print(f"Stylized video saved to: {output_path}")
video_path = r"C:\Users\ratna\Videos\DemoVideo.mp4"
style_image_path = r"C:\Users\ratna\Lostart\starry-night.jpg"
output_path = "output_video.mp4"
video = VideoFileClip(video_path)
video.write_videofile(output_path, codec='libx264',audio=False)
print(f"Video saved to {output_path}")
model_path = r"C:\Users\ratna\Lostart\arbitrary-image-stylization-v1-tensorflow1-256-v2"
num_frames = 10  # Process only the first 100 frames

apply_style_to_video(video_path, style_image_path, output_path, model_path, num_frames=num_frames)
