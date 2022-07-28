# Import Trained Model 

import tensorflow as tf
import numpy as np 
import pandas as pd
import streamlit as st 
from functions import parse_function, create_dataset, triage_classifier

# Import Model 
model = tf.keras.models.load_model('../../b_model.h5')


# Define Disease labels 
labels = ['Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Effusion',
 'Emphysema',
 'Fibrosis',
 'Infiltration',
 'Mass',
 'Nodule',
 'Pleural_Thickening',
 'Pneumonia',
 'Pneumothorax']

## Perform Operations on Streamlit 

# Write text 
st.title("Chest X-Ray Multi-Label Image Classification")
st.header("This is a multi-label image classification web-app designed to identify 13 diseases in chest X-ray images.")
uploaded_files = st.file_uploader("Please upload image file(s)", type=["jpg", "png"], accept_multiple_files= True)


# Define the threshold for positive hits 
threshold = st.slider('Set threshold for positive result', 0, 100, 20)
st.write('Current probability threshold is ', threshold, '%')


