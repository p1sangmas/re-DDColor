import streamlit as st
import subprocess
from PIL import Image

# Title of the app
st.title("Grayscale Image Colorization")

# File uploader
uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded image to a temporary file
    input_path = "./assets/test_images/uploaded_image.JPEG"
    image.save(input_path)

    # Button to run the colorization
    if st.button("Colorize"):
        # Run the colorization script
        output_path = "./colorize_output/colorized_image.png"
        command = [
            "CUDA_VISIBLE_DEVICES=0",
            "python3", "inference/colorization_pipeline.py",
            "--input_file", input_path,
            "--output_file", output_path,
            "--model_path", "modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
        ]
        subprocess.run(" ".join(command), shell=True)

        # Display the colorized image
        colorized_image = Image.open(output_path)
        st.image(colorized_image, caption='Colorized Image', use_column_width=True)
