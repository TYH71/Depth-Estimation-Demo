# import libraries/dependencies/modules
import streamlit as st
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from PIL import Image
import time
import datetime

# @st.cache()
def load_model(path: str = "asset/MDE_256x256_0004_50_16"):
    model = keras.models.load_model(path)
    return model

# @st.cache()
def load_demo_image(path: str = "asset/*.jpg"):
    # Prepare Sample Image Data
    images = []
    for f in glob.glob("./asset/image/*.jpg"):
        images.append(np.asarray(Image.open(f).convert("RGB").resize((256, 256)), dtype=np.float32)/255)
    images = np.array(images) # Shape: (None, 256, 256, 3)
    return images

if __name__ == '__main__':
    st.title("Monocular Depth Estimation")
    st.info("Monocular Depth Estimation is the task of estimating the depth value (distance relative to the camera) of each pixel given a single (monocular) RGB image.")
    
    col1, col2 = st.columns(2)
    
    # Sidebar - Receive Input Images
    sidebar = st.sidebar
    sidebar.header("Upload own Image")
    file = sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)
    sidebar.header("Demo Images")
    demo = sidebar.button("Demo")

    # Load MDE Model
    model = load_model()

    if demo:
        images = load_demo_image()
        rand_idx = np.random.randint(0, len(images))
        img = images[rand_idx]
        
        # Column 1 - Displaying Input Image
        col1.subheader("Input Image")
        col1.image(img, use_column_width=True)

        # Sample Inference
        img = img.reshape(1, 256, 256, 3)
        col1.write("Running Inference...")
        print("[{}]".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) # for logging
        start = time.time()
        pred = model.predict(img) # Shape: (None, 256, 256, 1)
        pred = pred[0].squeeze()
        end = time.time()
        inference_time = end - start
        
        # Column 2 - Display Predicted Depth Map
        col2.subheader("Predicted Depth Map")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        ax.imshow(pred, cmap="jet")
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        col2.pyplot(fig, caption='Predicted Depth Map') 
        col2.write("Inference Time: {:.3f}s".format(inference_time))
        

    if file:
        img = Image.open(file).convert("RGB").resize((256, 256))
        img = np.asarray(img, dtype=np.float32)/255
        img = img.reshape(1, 256, 256, 3)

        # Column 1 - Displaying Input Image
        col1.subheader("Input Image")
        col1.image(file, use_column_width=True)

        # Sample Inference
        col1.write("Running Inference...")
        print("[{}]".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) # for logging
        start = time.time()
        pred = model.predict(img) # Shape: (None, 256, 256, 1)
        pred = pred[0].squeeze()
        end = time.time()
        inference_time = end - start
        
        # Column 2 - Display Predicted Depth Map
        col2.subheader("Predicted Depth Map")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        ax.imshow(pred, cmap="jet")
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        col2.pyplot(fig, caption='Predicted Depth Map') 
        col2.write("Inference Time: {:.3f}s".format(inference_time))
        
    