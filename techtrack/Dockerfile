# Use the official Python image with OpenCV support
FROM python:3.8-slim-buster

# Install necessary packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*




# Install OpenCV with FFmpeg support
RUN pip install opencv-python-headless numpy

# Install any additional Python dependencies
# For example, if you have a requirements.txt file
COPY requirements.txt .
RUN pip install -r requirements.txt


# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# # Copy the YOLO model files

# COPY yolo_model_2 /app/yolo_model_2

# Expose any necessary ports (if applicable)
# EXPOSE 5000

# Command to run your inference service
CMD ["python", "inference_service.py"]
