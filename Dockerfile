# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies
# UI
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    && apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add API Call method
