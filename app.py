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

# Cache the model loading function
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
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

# Cache the model loading function
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
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

# Custom CSS for Sidebar Styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .sidebar .stSelectbox label {
        font-weight: bold;
        color: #333;
    }
    .sidebar .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 5px;
    }
    .sidebar .stSelectbox div[data-baseweb="select"] input {
        caret-color: transparent; /* Removes the blinking cursor */
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("🔍 Neural Style Transfer Studio")
st.sidebar.markdown("### Choose a section:")
page = st.sidebar.selectbox("Select a section", ["Home", "TensorFlow", "CNN VGG19", "Video Style Transfer"], index=0)

if page == "Home":
    st.title("🎨 Neural Style Transfer Studio")
    
    # Add hero section with description
    st.markdown("""
        <div style='background-color: #2E303E; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h2 style='color: #ff4b4b; margin-bottom: 1rem;'>Transform Your Images with AI</h2>
            <p style='color: #ffffff; font-size: 1.1rem; line-height: 1.6;'>
                Experience the magic of neural style transfer - an advanced AI technique that reimagines your photos in the style of any artwork. Our platform combines the power of deep learning with an intuitive interface to help you create stunning artistic transformations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.header("🚀 Available Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background-color: #343644; padding: 1.5rem; border-radius: 10px; height: 100%;'>
                <h3 style='color: #ff4b4b;'>🖼️ TensorFlow Style Transfer</h3>
                <p style='color: #ffffff;'>
                    Use Google's TensorFlow-based model for quick and efficient style transfer. Perfect for:
                    - Instant results
                    - High-quality transformations
                    - Multiple style experiments
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #343644; padding: 1.5rem; border-radius: 10px; height: 100%;'>
                <h3 style='color: #ff4b4b;'>🎯 VGG19 Neural Network</h3>
                <p style='color: #ffffff;'>
                    Leverage the power of VGG19 for detailed style transfer. Ideal for:
                    - Fine-tuned control
                    - Custom iterations
                    - Advanced style manipulation
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.header("🔍 How It Works")
    st.markdown("""
        <div style='background-color: #2E303E; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: #ff4b4b; margin-bottom: 1rem;'>Simple Steps to Transform Your Images</h3>
            <ol style='color: #ffffff; font-size: 1.1rem; line-height: 1.6;'>
                <li><strong>Choose Your Method:</strong> Select either TensorFlow or VGG19 from the navigation menu</li>
                <li><strong>Upload Images:</strong> Provide a content image (your photo) and a style image (the artistic style)</li>
                <li><strong>Process:</strong> Let our AI work its magic to combine both images</li>
                <li><strong>Download:</strong> Save your uniquely styled creation</li>
            </ol>
        </div>
        
        <div style='background-color: #343644; padding: 2rem; border-radius: 10px;'>
            <h3 style='color: #ff4b4b; margin-bottom: 1rem;'>✨ Pro Tips</h3>
            <ul style='color: #ffffff; font-size: 1.1rem; line-height: 1.6;'>
                <li>For best results, use high-quality images with clear subjects</li>
                <li>Experiment with different style images to find the perfect match</li>
                <li>Try both TensorFlow and VGG19 methods to compare results</li>
                <li>Use the video feature to apply styles to your video content</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif page == "TensorFlow":
    st.title("TensorFlow")
    st.header("Upload Images")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        content_image = load_image(content_file)
        style_image = load_image(style_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Content Image")
            st.image(np.squeeze(content_image), use_container_width=True)
        with col2:
            st.subheader("Style Image")
            st.image(np.squeeze(style_image), use_container_width=True)

        st.subheader("Stylized Image")
        with st.spinner("Applying style transfer..."):
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

        stylized_image = np.squeeze(stylized_image)
        stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0).numpy()

        stylized_image_pil = Image.fromarray((stylized_image * 255).astype(np.uint8))
        st.image(stylized_image, use_container_width=True, caption="Stylized Image")

        st.markdown(get_image_download_link(stylized_image_pil, "stylized_image.jpg", "📥 Download Stylized Image"), unsafe_allow_html=True)

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

        start_time = time.time()
        nst_vgg19.NUM_ITER = epochs
        nst_vgg19.main(content_path, style_path)
        end_time = time.time()
        execution_time = end_time - start_time
        
        st.success(f"Neural Style Transfer completed in {execution_time:.2f} seconds!")

elif page == "Video Style Transfer":
    st.title("🎥 Neural Style Transfer for Video")
    st.header("Upload Video and Style Image")

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    max_frames_to_process = st.number_input("Number of Frames to Process", min_value=1, step=1, value=5)

    if video_file and style_image_file:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        style_image_path = "uploaded_style.jpg"
        with open(style_image_path, "wb") as f:
            f.write(style_image_file.getbuffer())

        if st.button("Start Video Style Transfer"):
            output_frames_folder = "processed_frames"
            if not os.path.exists(output_frames_folder):
                os.makedirs(output_frames_folder)

            frame_count = extract_frames("uploaded_video.mp4", output_frames_folder, max_frames=max_frames_to_process)
            style_image = load_image(style_image_file)

            st.write(f"Processing {frame_count} frames...")
            progress_bar = st.progress(0)
            styled_frame_paths = []

            for i in range(frame_count):
                frame_path = os.path.join(output_frames_folder, f"frame_{i:04d}.jpg")
                frame_image = tf.convert_to_tensor(cv2.imread(frame_path)[..., ::-1], dtype=tf.float32)
                frame_image = frame_image[tf.newaxis, ...] / 255.0

                stylized_frame = model(frame_image, style_image)[0]
                stylized_frame = np.squeeze(stylized_frame)
                stylized_frame = tf.clip_by_value(stylized_frame, 0.0, 1.0).numpy()

                output_frame_path = os.path.join(output_frames_folder, f"styled_frame_{i:04d}.jpg")
                Image.fromarray((stylized_frame * 255).astype(np.uint8)).save(output_frame_path)
                styled_frame_paths.append(output_frame_path)
                
                progress_bar.progress((i + 1) / frame_count)

            # Save final video
            output_video_path = "styled_video.mp4"
            save_video(output_frames_folder, output_video_path)

            st.success("Video Style Transfer Complete!")
            
            # Display all frames
            st.subheader("All Styled Frames")
            for idx, frame_path in enumerate(styled_frame_paths):
                st.image(frame_path, caption=f"Frame {idx+1}")