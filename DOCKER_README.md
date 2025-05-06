# Docker Setup for DDColor

This document describes how to run the DDColor application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Getting Started

### 1. Download the pre-trained model

Before building the Docker image, make sure the pre-trained model is available:

```bash
python pretraindownload.py
```

This will download the model to the `modelscope/damo/cv_ddcolor_image-colorization` directory.

### 2. Build and run with Docker Compose

```bash
docker-compose up -d
```

This command will:
- Build the Docker image with all necessary dependencies
- Start the Streamlit web application in a container
- Map port 8501 from the container to your host

### 3. Access the application

Open your browser and navigate to:
```
http://localhost:8501
```

### 4. Stop the application

```bash
docker-compose down
```

## Manual Docker Build and Run

If you prefer to use Docker commands directly:

```bash
# Build the image
docker build -t ddcolor-app .

# Run the container
docker run -p 8501:8501 -v $(pwd)/modelscope:/app/modelscope ddcolor-app
```

## Troubleshooting

- If you encounter permission issues with the mounted volumes, check that your local user has proper access rights to the modelscope directory.
- For GPU support, you would need to install the NVIDIA Container Toolkit and add the appropriate flags to your docker run command or docker-compose file.