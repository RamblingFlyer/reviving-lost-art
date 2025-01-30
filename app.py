import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
import os
import shutil
import tempfile
import importlib.util
import time

# Function to clear the TensorFlow Hub cache
def clear_tfhub_cache():
    cache_dir = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

# Clear the TensorFlow Hub cache
clear_tfhub_cache()

# Load the Magenta model
@st.cache_resource
def load_magenta_model():
    model_path = r'C:\Users\ratna\Lostart\arbitrary-image-stylization-v1-tensorflow1-256-v2'
    return tf.saved_model.load(model_path)

# Load the model
try:
    magenta_model = load_magenta_model()
except Exception as e:
    st.error(f"Error loading the Magenta model: {e}")
    st.stop()

# Helper functions for image processing
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0  # Normalize to [0, 1]
    img = img[tf.newaxis, :]
    return img

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Video processing helper functions
def extract_frames(video_path, output_folder, max_frames=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    cap.release()
    return frame_count

def save_video(frame_folder, output_video_path, fps=30):
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #21222E;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "TensorFlow", "CNN VGG19", "Video Style Transfer"])

if page == "Home":
    st.title("🎨 Neural Style Transfer")
    st.write(
        """
        Neural style transfer is a technique for generating images that combine the content of one image with the style of another image using a convolutional neural network. 
        Instructions:
        1. Go to the respective pages to upload your content and style images or videos.
        2. Wait for the model to process the input and generate a stylized output.
        3. Download the resulting stylized media.
        Note: Processing time may vary depending on input size.
        """
    )

elif page == "TensorFlow":
    st.title("TensorFlow")
    st.header("Upload Images")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        content_image = load_image(content_file)
        style_image = load_image(style_file)

        st.subheader("Stylized Image")
        with st.spinner("Applying style transfer..."):
            stylized_image = magenta_model(tf.constant(content_image), tf.constant(style_image))[0]

        stylized_image = np.squeeze(stylized_image)
        stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0).numpy()

        stylized_image_pil = Image.fromarray((stylized_image * 255).astype(np.uint8))
        st.image(stylized_image, use_container_width=True, caption="Stylized Image")

        st.markdown(get_image_download_link(stylized_image_pil, "stylized_image.jpg", "💾 Download Stylized Image"), unsafe_allow_html=True)

elif page == "CNN VGG19":
    st.title("CNN VGG19")
    st.header("Run VGG19 Neural Style Transfer")
    content_file_vgg19 = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content_vgg19")
    style_file_vgg19 = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style_vgg19")
    epochs = st.number_input("Enter the number of epochs", min_value=1, max_value=1000, value=1, step=1)

    if content_file_vgg19 and style_file_vgg19 and st.button("Start Style Transfer"):
        content_path = "content_vgg19.jpg"
        style_path = "style_vgg19.jpg"
        with open(content_path, "wb") as f:
            f.write(content_file_vgg19.getbuffer())
        with open(style_path, "wb") as f:
            f.write(style_file_vgg19.getbuffer())

        spec = importlib.util.spec_from_file_location("neural_style_transfer_vgg19", "neural_style_transfer_vgg19.py")
        nst_vgg19 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nst_vgg19)

        nst_vgg19.NUM_ITER = epochs
        nst_vgg19.main(content_path, style_path)


elif page == "Video Style Transfer":
    st.title("Video Style Transfer")
    st.header("Upload Video and Style Image")

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video")
    style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style_image")
    max_frames_to_process = st.number_input("Number of Frames to Process", min_value=1, step=1, value=5)

    if video_file and style_image_file:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        style_image_path = "uploaded_style.jpg"
        with open(style_image_path, "wb") as f:
            f.write(style_image_file.getbuffer())

        if st.button("Start Video Style Transfer"):
            output_frames_folder = "processed_frames"

            frame_count = extract_frames("uploaded_video.mp4", output_frames_folder, max_frames=max_frames_to_process)
            style_image = load_image(style_image_file)

            st.write(f"Processing {frame_count} frames...")
            styled_frame_paths = []

            for i in range(frame_count):
                frame_path = os.path.join(output_frames_folder, f"frame_{i:04d}.jpg")
                frame_image = load_image(frame_path)

                with st.spinner(f"Processing frame {i + 1} / {frame_count}..."):
                    stylized_frame = magenta_model(tf.constant(frame_image), tf.constant(style_image))[0]
                    stylized_frame = np.squeeze(stylized_frame)
                    stylized_frame = tf.clip_by_value(stylized_frame, 0.0, 1.0).numpy()

                    output_frame_path = os.path.join(output_frames_folder, f"styled_frame_{i:04d}.jpg")
                    Image.fromarray((stylized_frame * 255).astype(np.uint8)).save(output_frame_path)
                    styled_frame_paths.append(output_frame_path)

            # Display styled frames on the Streamlit page
            st.subheader("Styled Frames")
            styled_images = [Image.open(frame_path) for frame_path in styled_frame_paths]
            for i, img in enumerate(styled_images):
                st.image(img, caption=f"Styled Frame {i + 1}", use_container_width=True)

            # Save final video
            output_video_path = "styled_video.mp4"
            save_video(output_frames_folder, output_video_path)

            st.success("Video Style Transfer Complete!")
