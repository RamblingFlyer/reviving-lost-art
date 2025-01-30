import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model
from PIL import Image
import cv2

# Constants
RESIZE_HEIGHT = 480  # Adjust for your video's resolution
NUM_ITER = 100  # Number of iterations per frame
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1e-1
CONTENT_LAYER_NAME = "block5_conv2"
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# Helper functions
def preprocess_image(image_path, target_height, target_width):
    img = Image.open(image_path).resize((target_width, target_height))
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = vgg19.preprocess_input(img_array)
    return tf.convert_to_tensor(img_array)

def deprocess_image(tensor, target_height, target_width):
    tensor = tensor.numpy()
    tensor = tensor.reshape((target_height, target_width, 3))
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.680
    tensor = tensor[:, :, ::-1]  # Convert BGR to RGB
    return np.clip(tensor, 0, 255).astype("uint8")

def gram_matrix(tensor):
    features = tf.reshape(tensor, (-1, tensor.shape[-1]))
    gram = tf.matmul(tf.transpose(features), features)
    return gram

def compute_loss(feature_extractor, combination_image, content_features, style_features):
    combination_features = feature_extractor(combination_image)
    content_loss = tf.reduce_sum(
        tf.square(combination_features[CONTENT_LAYER_NAME] - content_features[CONTENT_LAYER_NAME])
    ) / 2
    style_loss = 0
    for layer_name in STYLE_LAYER_NAMES:
        style_gram = gram_matrix(style_features[layer_name][0])
        combination_gram = gram_matrix(combination_features[layer_name][0])
        channels = combination_features[layer_name].shape[-1]
        style_loss += tf.reduce_sum(tf.square(style_gram - combination_gram)) / (4.0 * (channels ** 2) * combination_features[layer_name].shape[1] ** 2)
    return CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

def extract_frames(video_path, output_folder, max_frames=5):
    """Extract frames from the video, up to a maximum of `max_frames`."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:  # Limit the number of frames processed
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

def save_video(frame_folder, output_video_path, fps=30):
    """Save the processed frames back as a video."""
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    print(f"Video saved to '{output_video_path}'.")

def apply_style_transfer(content_image_path, style_image_path, output_folder, height, width):
    """Apply style transfer to a single image."""
    # Preprocessing
    content_tensor = preprocess_image(content_image_path, height, width)
    style_tensor = preprocess_image(style_image_path, height, width)
    generated_image = tf.Variable(tf.random.uniform(content_tensor.shape, dtype=tf.float32))

    # Model setup
    vgg_model = vgg19.VGG19(weights="imagenet", include_top=False)
    output_layers = {layer.name: layer.output for layer in vgg_model.layers}
    feature_extractor = Model(inputs=vgg_model.inputs, outputs=output_layers)
    content_features = feature_extractor(content_tensor)
    style_features = feature_extractor(style_tensor)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)

    # Style transfer
    for iteration in range(NUM_ITER):
        with tf.GradientTape() as tape:
            loss = compute_loss(feature_extractor, generated_image, content_features, style_features)
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        if (iteration + 1) % 10 == 0:  # Progress logging every 10 iterations
            print(f"Iteration {iteration + 1}/{NUM_ITER}, Loss: {loss.numpy():.4f}")
    print(f"Style transfer completed for '{content_image_path}'.")
    return deprocess_image(generated_image, height, width)

def process_video(video_path, style_image_path, output_video_path, max_frames=5):
    """Process a video with style transfer, limited to `max_frames` for testing."""
    output_frames_folder = "extracted_frames"
    styled_frames_folder = "styled_frames"
    extract_frames(video_path, output_frames_folder, max_frames)

    frame_paths = sorted([os.path.join(output_frames_folder, f) for f in os.listdir(output_frames_folder) if f.endswith('.jpg')])
    first_frame = Image.open(frame_paths[0])
    width, height = first_frame.size

    if not os.path.exists(styled_frames_folder):
        os.makedirs(styled_frames_folder)

    for i, frame_path in enumerate(frame_paths):
        print(f"Processing frame {i + 1}/{len(frame_paths)}...")
        styled_frame = apply_style_transfer(frame_path, style_image_path, styled_frames_folder, height, width)
        styled_frame_path = os.path.join(styled_frames_folder, f"styled_frame_{i:04d}.jpg")
        Image.fromarray(styled_frame).save(styled_frame_path)

    save_video(styled_frames_folder, output_video_path)
    print("Video processing completed.")

if __name__ == "__main__":
    # Paths
    input_video_path = r"C:\Users\ratna\Lostart\VID20241003115459.mp4"
    style_image_path = r"C:\Users\ratna\Lostart\starry-night.jpg"
    output_video_path = r"C:\Users\ratna\Lostart\styled_video.mp4"

    process_video(input_video_path, style_image_path, output_video_path, max_frames=5)
