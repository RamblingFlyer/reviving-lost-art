
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
    if (os.path.exists(cache_dir)):
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

# Function to load and preprocess the image
def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0  # Normalize to [0, 1]
    img = img[tf.newaxis, :]
    return img

# Function to generate download link for the image
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

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

# Page selection
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "TensorFlow", "CNN VGG19"])

if page == "Home":
    st.title("🎨 Neural Style Transfer")
    st.write(
        """
        Neural style transfer is a technique for generating images that combine the content of one image with the style of another image using a convolutional neural network. 
        This technique involves separating and recombining the content and style of images using deep learning models.
        
        Instructions:
        1. Go to the "Neural Style Transfer Info 1" page to upload your content and style images.
        2. Wait for the model to process the images and generate a stylized output.
        3. Download the resulting image from the same page.
        
        Note: The processing time may vary depending on the size of the images.
        """
    )
elif page == "TensorFlow":
    st.title("TensorFlow")
    st.write("Upload a content image and a style image, and watch the neural network transform your content image using the style of the style image!")

    # Image upload section
    st.header("Upload Images")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        st.write("Both images uploaded successfully!")

        # Layout for images
        col1, col2 = st.columns(2)

        # Load and display content image
        content_image = load_image(content_file)
        with col1:
            st.subheader("Content Image")
            st.image(np.squeeze(content_image), use_column_width=True, caption="This is the content image.")

        # Load and display style image
        style_image = load_image(style_file)
        with col2:
            st.subheader("Style Image")
            st.image(np.squeeze(style_image), use_column_width=True, caption="This is the style image.")

        # Perform style transfer with progress bar
        st.subheader("Stylized Image")
        with st.spinner("Applying style transfer..."):
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

        # Normalize and clip the stylized image to ensure it's within the [0.0, 1.0] range
        stylized_image = np.squeeze(stylized_image)
        stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0).numpy()

        # Convert stylized image to PIL format for download link
        stylized_image_pil = Image.fromarray((stylized_image * 255).astype(np.uint8))

        # Display stylized image
        st.image(stylized_image, use_column_width=True, clamp=True, caption="This is the stylized image.")

        # Provide download link
        st.markdown(get_image_download_link(stylized_image_pil, "stylized_image.jpg", "📥 Download Stylized Image"), unsafe_allow_html=True)

        # Save stylized image
        result_image_path = "generated_img.jpg"
        cv2.imwrite(result_image_path, cv2.cvtColor(stylized_image * 255, cv2.COLOR_RGB2BGR))
        st.success("Stylized image saved as generated_img.jpg")
    else:
        st.write("Please upload both content and style images to proceed.")

    # Additional information section
    with st.expander("ℹ More Information"):
        st.write(
            """
            Neural style transfer is a technique for generating images that combine the content of one image with the style of another image using a convolutional neural network. 
            This app uses a pre-trained model from TensorFlow Hub to apply the style of the style image to the content image.
            
            Instructions:
            1. Upload your content image and style image using the sidebar.
            2. Wait for the model to process the images.
            3. Download the resulting stylized image using the provided link.
            
            Note: The processing time may vary depending on the size of the images.
            """
        )
elif page == "CNN VGG19":
    st.title("CNN VGG19")
    

    st.header("Run VGG19 Neural Style Transfer")
    st.write("Upload a content image and a style image, and watch the neural network transform your content image using the style of the style image!")

    # Image upload section for VGG19 Neural Style Transfer
    content_file_vgg19 = st.file_uploader("Upload Content Image for VGG19", type=["jpg", "jpeg", "png"], key="content_vgg19")
    style_file_vgg19 = st.file_uploader("Upload Style Image for VGG19", type=["jpg", "jpeg", "png"], key="style_vgg19")

    # Epochs input
    epochs = st.number_input("Enter the number of epochs", min_value=1, max_value=1000, value=1, step=1)

    if content_file_vgg19 and style_file_vgg19:
        st.write("Both images uploaded successfully!")

        # Load and save images to disk
        content_image_path_vgg19 = "content_vgg19.jpg"
        style_image_path_vgg19 = "style_vgg19.jpg"
        with open(content_image_path_vgg19, "wb") as f:
            f.write(content_file_vgg19.getbuffer())
        with open(style_image_path_vgg19, "wb") as f:
            f.write(style_file_vgg19.getbuffer())

        # Button to start processing
        if st.button("Start Style Transfer"):
            # Import and run the VGG19 neural style transfer script
            spec = importlib.util.spec_from_file_location("neural_style_transfer_vgg19", "neural_style_transfer_vgg19.py")
            nst_vgg19 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nst_vgg19)

            # Update NUM_ITER and run the script
            start_time = time.time()
            nst_vgg19.NUM_ITER = epochs
            nst_vgg19.main(content_image_path_vgg19, style_image_path_vgg19)
            end_time = time.time()
            execution_time = end_time - start_time
            time_per_epoch = execution_time / epochs

            # Predefined benchmark time (in seconds) for efficiency calculation
            benchmark_time = 30000  # Adjust this based on your requirements
            percentage_efficiency = (execution_time / benchmark_time) * 100

            st.success(f"Neural Style Transfer completed in {execution_time:.2f} seconds. Average time per epoch: {time_per_epoch:.2f} seconds. Efficiency: {percentage_efficiency:.2f}%")
    else:
        st.write("Please upload both content and style images to proceed.")
else:
    st.write("Page not found!")
