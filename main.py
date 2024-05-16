import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
import json
import time
import pandas as pd

# SAVE THE FRAME TO A FILE
def save_frames_to_files(frames, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame as an image file
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image.save(os.path.join(output_dir, f"frame_{i}.jpg"))

# DEFAULT PRINTS
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# DEFINE THE MODEL
def build_model(seq_length, img_height, img_width):
    model = Sequential([
        TimeDistributed(Flatten(), input_shape=(seq_length, img_height, img_width, 3)),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LOAD IMAGES FROM FOLDERS (FOR TEACHING THE MODEL)
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            # Assuming folder names indicate labels (camera_on/camera_off)
            labels.append(1 if folder == 'C:/Users/Lenovo/Desktop/LSTM/camera_on' else 0)
    return images, labels

# PREPROCESS IMAGES
def preprocess_images(images):
    # Resize images to a common size (e.g., 64x64)
    resized_images = [cv2.resize(img, (64, 64)) for img in images]
    # Normalize pixel values (optional)
    normalized_images = [img / 255.0 for img in resized_images]
    return np.array(normalized_images)

# Generate dummy data
num_samples = 1000
seq_length = 10  # Number of frames per sequence
img_height = 64
img_width = 64
frames = []

# READ MJPEG FEED FROM 
def read_mjpeg_feed(url):
    try:
        # Create VideoCapture object with MJPEG URL
        cap = cv2.VideoCapture(url)
        
        for i in range(10):
            # Read frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break

            frames.append(frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the VideoCapture object and close the OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print("Error reading MJPEG feed:", e)

#READ ONE MJPEG
def read_mjpeg(url):
    try:
        # Create VideoCapture object with MJPEG URL
        cap = cv2.VideoCapture(url)
        
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            return
        
        cap.release()
        cv2.destroyAllWindows()
        return frame
    
    except Exception as e:
        print("Error reading MJPEG feed:", e)

# Example usage:
url = "https://vrellab.zmitac.aei.polsl.pl:10114/"  # Replace this URL with the actual MJPEG stream URL

base_url= "https://vrellab.zmitac.aei.polsl.pl:101"
urls=[]
for i in range(16):
    if i+1 < 10:
        urls.append(base_url + "0" + str(i+1) + "/")
    else:
        urls.append(base_url + str(i+1) + "/")
print(urls)

#read_mjpeg_feed(url)
#print(frames)

def display_images(images, titles=None):
    num_images = len(images)
    
    # Create subplots with appropriate number of rows and columns
    rows = (num_images + 1) // 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')  # Assuming grayscale images
            ax.axis('off')
            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')  # Hide empty subplot
    
    plt.tight_layout()
    plt.show()

#display_images(frames)
#save_frames_to_files(frames, "C:/Users/Lenovo/Desktop/LSTM/test")

# TEACHING THE MODEL

camera_on_images, camera_on_labels = load_images_from_folder('C:/Users/Lenovo/Desktop/LSTM/camera_on')
print(camera_on_images[0].shape)
camera_off_images, camera_off_labels = load_images_from_folder('C:/Users/Lenovo/Desktop/LSTM/camera_off')
camera_test_images, camera_test_labels = load_images_from_folder('C:/Users/Lenovo/Desktop/LSTM/test')

all_images = np.concatenate([camera_on_images, camera_off_images], axis=0)
all_labels = np.concatenate([camera_on_labels, camera_off_labels], axis=0)

print("camera on labels", camera_on_labels)
processed_images = preprocess_images(all_images)
process_test_images = preprocess_images(camera_test_images)

print(process_test_images[0].shape)
print(len(process_test_images))
# Shuffle data (optional)
train_images, test_images, train_labels, test_labels = train_test_split(processed_images, all_labels, test_size=0.2, random_state=42)

# Define model architecture
seq_length = processed_images.shape[1]  # Number of frames per sequence
img_height = processed_images.shape[2]
img_width = processed_images.shape[3]

model = Sequential([
    TimeDistributed(Flatten(), input_shape=(seq_length, img_height, img_width)),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("train_labels:", train_labels)

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss, accuracy)

predictions = []

while(True):
    frames=[]
    for url in urls:
        frame = read_mjpeg(url)
        if frame is not None:    
            print(url)
            frames.append(frame)
        else:
            frames.append(read_mjpeg("C:/Users/Lenovo/Desktop/LSTM/camera_off/frame_0.jpg"))
    print(type(frames))
    processed_frame = preprocess_images(frames)
    predictions = model.predict(processed_frame)
    labels = np.round(predictions).astype(int)
    with open("predictions.json", "w") as file:
    # Loop through the list and write each item to the file
        cameras = []
        for i in range(16):
            cameras.append("cam" + str(i+1))
        print(labels)
        labels = labels.tolist()
        data_dict = {cameras[i]: labels[i][0] for i in range(len(labels))}
        file.write(json.dumps(data_dict)) 
    time.sleep(30)

# Evaluate the model (optional)
# X_test, y_test = generate_dummy_data(num_samples, seq_length, img_height, img_width)
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# Once the model is trained, you can use it for inference on new data
# For example, you can pass a new sequence of frames through the model
# and observe the output predictions to detect when the camera is turned off.