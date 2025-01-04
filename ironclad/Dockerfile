# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Define environment variables
ENV MODEL_NAME=casia-webface
ENV DEVICE=cpu
ENV CATALOG_DIR=/app/storage/catalog

# Run the Flask server
CMD ["python", "app.py"]
