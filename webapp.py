import streamlit as st
import subprocess
from PIL import Image
import os

# Title of the app
st.title("Grayscale Image Colorization")

# File uploader
uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Create directories if they don't exist
    os.makedirs("./assets/test_images", exist_ok=True)
    os.makedirs("./colorize_output", exist_ok=True)

    # Save the uploaded image to a temporary file
    input_path = "./assets/test_images/uploaded_image.JPEG"
    image.save(input_path)

    # Button to run the colorization
    if st.button("Colorize"):
        with st.spinner("Colorizing your image..."):
            # Run the colorization script
            output_path = "./colorize_output/colorized_image.png"
            model_path = "modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
            
            # Check if model exists
            if not os.path.exists(model_path):
                st.warning("Model not found. Downloading model...")
                subprocess.run("python pretraindownload.py", shell=True)
            
            # Run the colorization pipeline
            command = [
                "python", "inference/colorization_pipeline.py",
                "--input_file", input_path,
                "--output_file", output_path,
                "--model_path", model_path
            ]
            
            # Remove CUDA_VISIBLE_DEVICES to use appropriate device automatically
            result = subprocess.run(" ".join(command), shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                st.error(f"Colorization failed: {result.stderr}")
            else:
                # Display the colorized image
                if os.path.exists(output_path):
                    colorized_image = Image.open(output_path)
                    st.image(colorized_image, caption='Colorized Image', use_column_width=True)
                    st.success("Image colorized successfully!")
                    st.download_button(
                        label="Download colorized image",
                        data=open(output_path, "rb").read(),
                        file_name="colorized_image.png",
                        mime="image/png"
                    )
                else:
                    st.error("Colorization completed but output file not found.")
