
# Use a base image with CUDA and Python pre-installed
# Check NVIDIA NGC for up-to-date PyTorch containers or use official Python images with CUDA
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, git, and ffmpeg (for video processing utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep the image smaller
RUN pip3 install --no-cache-dir -r requirements.txt

# Optional: For potentially better performance on NVIDIA GPUs with compatible models
# RUN pip3 install --no-cache-dir xformers

# Set up the working directory
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Create directories for Hugging Face cache, input, and output
# This allows mounting volumes to these locations for persistence
RUN mkdir -p /app/huggingface_cache /app/input_data /app/output_data

# Set Hugging Face cache directory environment variable
ENV HF_HOME=/app/huggingface_cache
ENV XDG_CACHE_HOME=/app/huggingface_cache # Some older versions might use this

# Command to run your script
# The script will expect JSON in /app/input_data and write to /app/output_data
ENTRYPOINT ["python3", "process_video_batch.py"]
# Default CMD if no arguments are passed to `docker run my-video-processor`
CMD ["--json_file", "/app/input_data/input_scenes.json", "--output_dir", "/app/output_data"]
