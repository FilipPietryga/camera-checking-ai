# Use an official Python runtime as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install tensorflow-gpu==2.0.0
RUN pip install tensorflow_hub
RUN pip install tensorflow_datasets

# Run app.py when the container launches
CMD ["python", "main.py"]