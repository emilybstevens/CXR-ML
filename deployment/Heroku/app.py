### DEPENDENCY IMPORTS ###
from operator import iconcat
import streamlit as st 
import tensorflow as tf
import cv2 as cv
from PIL import Image, ImageOps 
import numpy as np 
from labels import disease_labels
import pandas as pd

### IMAGE ICON SET ###
icon = Image.open('Resources/logo-dark.png')
st.set_page_config(page_title = 'X-Radar Net', page_icon = icon)

### MODEL IMPORT ###
# Import Model store it in cache as a function to prevent reloading 
st.set_option('deprecation.showfileUploaderEncoding', False) 
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('model/delta_model.h5')
    return model 

### LOAD IMAGE ###
def load_image():
    # Create Widget for Image File upload
    file = st.file_uploader(label="Please upload an chest x-ray", type=["jpg", "png"])
    if file is not None:
        st.write('Filename: ', file.name) 
        image = Image.open(file)
        st.image(image, use_column_width= True)
        return image
    else:
        return None

### DEFINE THRESHOLD ###
def thresholder():
    # Define the threshold for positive hits 
    threshold = st.slider('Set threshold for positive result', 0, 100, 20)
    st.write('Current probability threshold is ', threshold, '%')
    return threshold

### DEPLOY MODEL AND PREDICT ###
def import_and_predict(image_data, model, labels):
    size = (224, 224) # Define size required for the model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS) # Resize Image
    img = np.asarray(image) # Convert to NP array
    color_img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Ensure the color channels are preserved
    resize_img = color_img/225 # Normalize the image
    img_reshape = resize_img[np.newaxis,...] # Make the Image 4D for Model Compatibility
    prediction = model.predict(img_reshape) # Make Predictions of the image
    return prediction 

### POSITIVE BINARY LABEL CREATION ###
def binary_classifier(prediction, threshold, dlabels):
    binary_class = (prediction > (threshold/100)).astype(int)
    results=pd.DataFrame(binary_class, columns=dlabels)
    return results 

### TRIAGE CLASSIFIER ###
def triage_classifier(df):
    copy_df = df.copy() # Creates a copy of the dataframe 
    copy_df['Triage'] = np.nan # Creates a new empty column for the triage assignment 
    for ind in copy_df.index: # loops through dataframe based on index
        # Emergent = Atelectasis, Consolidation, Edema, Effusion, Infiltration, Pneumothorax
        if (copy_df['Atelectasis'][ind] == 1)\
            or (copy_df['Consolidation'][ind] ==1)\
                or (copy_df['Edema'][ind] == 1)\
                    or (copy_df['Effusion'][ind] == 1)\
                        or (copy_df['Infiltration'][ind] == 1)\
                            or (copy_df['Pneumothorax'][ind] == 1):
                            copy_df['Triage'][ind] = 'Emergent'                    

        # Acute = Mass, Pneumonia. Hernia is ignored due to insufficient sample size.                    
        elif (copy_df['Mass'][ind] == 1) or (copy_df['Pneumonia'][ind] == 1):
            copy_df['Triage'][ind] = 'Acute'
            
        # Chronic = Cardiomegaly, Emphysema, Fibrosis, Nodule, Pleural Thickening 
        elif (copy_df['Cardiomegaly'][ind] == 1)\
            or (copy_df['Emphysema'][ind] ==1)\
                or (copy_df['Fibrosis'][ind] == 1)\
                    or (copy_df['Nodule'][ind] == 1)\
                        or (copy_df['Pleural_Thickening'][ind] == 1):
                        copy_df['Triage'][ind] = 'Chronic'
                        
        # If no hits, then 'No Finding' 
        else: 
            copy_df['Triage'][ind] = 'No Finding'
            

    return copy_df 

def main():
    ### WEBPAGE TEXT COMPONENTS ###
    # Write text 
    st.title("Chest X-Ray Multi-Label Image Classification")
    st.header("This multi-label image classification web-app is designed to identify 13 lung diseases in chest X-ray images.")
    model = load_model()
    image = load_image()
    thresh = thresholder()
    result = st.button('Analyze Image')
    if result: 
        pred = import_and_predict(image, model, disease_labels)
        binary_df = binary_classifier(pred, thresh, disease_labels)
        triaged_df = triage_classifier(binary_df)
        st.dataframe(triaged_df)

# Run the main function. 
if __name__ == '__main__':
    main()









