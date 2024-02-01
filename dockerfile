# Start from the NVIDIA CUDA image with CUDA 12.3 support, based on Ubuntu 22.04
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the contents of your project into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Your command to run the application
CMD ["python3", "deploy.py"]
