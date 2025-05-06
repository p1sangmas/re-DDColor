FROM python:3.9-slim

WORKDIR /app

# Install system dependencies with comprehensive libraries for all dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch first - this is critical for colorization and for setup.py
RUN pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu

# Install scientific and image processing packages explicitly
# Install scipy before other packages that might depend on it
RUN pip install --no-cache-dir numpy==1.24.3 scipy==1.9.1 matplotlib

# Install other key dependencies explicitly
RUN pip install --no-cache-dir \
    timm==0.9.2 \
    Pillow==10.1.0 \
    PyYAML==6.0.1 \
    scikit-image==0.22.0 \
    opencv-python-headless \
    lmdb==1.4.1 \
    tqdm==4.65.0 \
    requests==2.31.0 \
    tensorboard \
    huggingface_hub \
    ipykernel \
    modelscope \
    streamlit

# Attempt to install dlib with fallback
RUN pip install --no-cache-dir dlib==19.24.1 --no-build-isolation || echo "Continuing without dlib"

# Create directories for images and models
RUN mkdir -p /app/assets/test_images
RUN mkdir -p /app/colorize_output
RUN mkdir -p /app/modelscope

# Copy the application code
COPY . .

# Create version.py file needed for setup.py installation
RUN python -c "import os; os.makedirs('basicsr', exist_ok=True); open('basicsr/version.py', 'w').write('__version__ = \"1.0.0\"\n__gitsha__ = \"unknown\"\nversion_info = (1, 0, 0)')"

# Install basicsr as a package with minimal dependencies
ENV BASICSR_EXT=False
RUN pip install -e . --no-dependencies

# Verify critical package installations
RUN python -c "import torch; print('PyTorch version:', torch.__version__)"
RUN python -c "import scipy; print('SciPy version:', scipy.__version__)"
RUN python -c "import timm; print('timm version:', timm.__version__)"
RUN python -c "import basicsr; print('basicsr package found')"

# Download model if it doesn't exist
RUN python pretraindownload.py

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variable to make Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Make sure CUDA device is properly handled in Docker
ENV CUDA_VISIBLE_DEVICES=""

# Command to run the application
CMD ["streamlit", "run", "webapp.py", "--server.port=8501", "--server.address=0.0.0.0"]