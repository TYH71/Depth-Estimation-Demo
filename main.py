import streamlit as st
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from PIL import Image

st.title("Monocular Depth Detection")

# Load MDE Model
model = keras.models.load_model("asset/DepthEstimationModel")


col1, col2 = st.columns(2)

# Prepare Sample Image Data
images = []
for f in glob.glob("./asset/image/*.jpg"):
    images.append(np.asarray(Image.open(f).resize((256, 256)), dtype=np.float32)/255)
images = np.array(images) # Shape: (None, 256, 256, 3)


# Sample Inference
pred = model.predict(images) # Shape: (None, 256, 256, 1)

fig, ax = plt.subplots(1, 1, figsize=(1, 1), tight_layout=True)
ax.imshow(pred[0].squeeze(), cmap="jet")
ax.axis('off')

col1.image(images[0], width=256, caption="Base Image")
col2.pyplot(fig, caption='Predicted Depth Map')
