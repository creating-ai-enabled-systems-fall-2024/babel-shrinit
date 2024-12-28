# base image
FROM python:3.9-slim

# Working directory
WORKDIR /app


COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libpq-dev \
    && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

# Flask app port
EXPOSE 5000

# Run 
CMD ["python", "pipeline.py"]
