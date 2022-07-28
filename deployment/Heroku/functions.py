# Import Image Processing Functions 
import numpy as np
import pandas as pd 
import tensorflow as tf


def parse_function(filename):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_resized, [224, 224])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized

# Define Constants for the image batcher function
BATCH_SIZE = 32 # Big enough to not crash the processor
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time

# Function to Generate the dataset required

def create_dataset(filenames):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

# Function to take in a dataframe of binary labels from CNN and assign Triage levels in a new column ['Triage']

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