import tensorflow as tf
model = tf.keras.models.load_model('../../b_model.h5')

import streamlit as st
st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )
st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2 
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")
    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)

# Define the threshold for positive hits 
threshold = st.slider('Set threshold for positive result', 0, 100, 20)
st.write('Current probability threshold is ', threshold, '%')