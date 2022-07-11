# Examining Chest X-Rays Through Machine Learning

## Our Topic

### Overview

To create a Machine Learning model that can analyze a patient's chest X-rays, 
classify lung disease, and flag X-rays displaying conditions with higher mortality rates to assist 
healthcare professionals in triaging high-risk patients faster. 

### Reasons for Selecting Topic

In recent years, the high rates of emergent lung disease have revealed a weakness in emergency medicine: emergency radiology. 
</br></br>
Traditionally, when a patient is admitted to the ER and has a chest X-ray performed (CXR), that CXR is entered into a waiting list to be read and interpreted by a radiologist. 
This interpretation is required for patients to receive any sort of active treatment or further hospitalization. 
From the time a CXR is taken, it can take several hours for the radiologist to return an interpretation to the ER staff, and without that interpretation, ER physicians cannot recommend any type of treatment for their patient. 
This means that while the ER physician is waiting for the radiologist's report, the patient is effectively on standby, regardless of the severity of their symptoms.
</br></br>
Unfortunately, most hospitals do not have radiology triage systems in place and as a result, emergent patients requiring immediate surgery will end up on the same waiting list as less dire patients, with no indicator 
way for radiologists to know which patient charts need to be prioritized. 
This lack of a useful CXR triage system results in longer healthcare wait times for emergent conditions requiring surgery, which can increase the mortality rate for those conditions. 
</br></br>
The goal of this model will be to create a machine learning model that can function as a triage system for this waiting list, flagging the higher-risk CXRs based on the presence of 
visual anomalies associated with higher-mortality conditions. With this triage system, radiologists can ensure that they are prioritizing the patients with potentially deadly conditions, 
reducing the rates of Adverse, Sentinel, and Never Events in ER settings by decreasing the amount of time from ER admittance to hospital admittance for emergent patients. 

## Project Outline
The scope of this project will be to create a prospective algorithm that can be marketed for use by ER staff for quick triage of CXRs, allowing emergent patients to be prioritized over non-emergent patients. 
A dashboard will be created that will include a visualization of what the application could look like once built out, an overview of the algorithm's current performance and accuracy, an overview of the data set 
used to train and test the algorithm, an overview of why the algorithm was created, and an about section detailing information about the team members. 

## Analysis
A machine learning model will undergo supervised training to understand how to identify various CXR diagnoses. The outputs of the trained model will be analyzed for precision and accuracy, as well as improvement over the original, provisional model. 
</br></br>
In addition, the testing/training data set will be analyzed for trends based on various factors, including gender, view position, age, disease level, and triage level. 
Dashboards will be created for each of these variables, and will be viewable within the overall site created for this project. 
Tableau will be used to initially investigate potential trends in the data set, and once the team has narrowed down which visuals to include, Javascript & Plotly will be used to visualize the data within the created site. 

## Our Data 

Data for this analysis will be sourced from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data?select=README_CHESTXRAY.pdf), utilizing chest X-rays compiled by the ``National Institutes of Health``. 
The dataset includes 112,120 X-ray images from 30,805 unique patients. </br></br>
**Citation**: *Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
IEEE CVPR 2017, ChestX-ray8Hospital-ScaleChestCVPR2017_paper.pdf*

## Our Questions 

* How can a Machine Learning Model Be Devised to Identify & Classify Lung Diseases in Chest X-Rays?
* How can we use our Machine Learning Model to increase efficiency in lung disease treatment?
* How can a Machine Learning Model Be Implemented in a Real-World Medical Scenario? 

## Google Slides

Our Presentation on Google Slides details the scope of our topic in greater depth, as well as illustrating why we chose our topic, and the exploratory and analytical phases we went through to create our Machine Learning Model. </br></br>
View [Presentation](https://docs.google.com/presentation/d/1rS79_7x5zY806TvwxHiqctWvBmpmMKyt_Wl2rek10Dc/edit#slide=id.p) </br></br>
Our Storyboard on Google Slides features a visual depiction of our preliminary dashboard, in addition to a comprehensive overview of said dashboard's tools and interactive elements. </br></br>
</br>View [storyboard](https://docs.google.com/presentation/d/1dvesALep-6vo8g94_rq3NUFCwnJmg3bc8BYD8B-T_Y4/edit)  

## Machine Learning Model

* Decsription of Preliminary Data Preprocessing

Since the dataset was large and relatively unbiased, very minimal pre-processing was required. The images had a consistent size of 1024 x 1024 that can be rescaled using keras' built in data generators. The 'Data_Entry_2017.csv' file had many columns that were unnecessary. The `Image Index` column was used to generate the relative paths of the images and the path values were used as the x_column input that helps the image data generator to find the image file in its respected nested folder. Although the data can be sorted into different folders based on its classes, around 20% of the images had multiple labels in the same image. Additionally, there were more than 100,000 images in total hence sorting into different training, testing folders was deemed inefficient. The features were located in the `Finding Labels` column. The rest of the columns were not discarded since we are unsure of its usage yet. 


* Description of preliminary feature engineering and preliminary feature selection, including their decision-making process

Since each sample could have up to 8 co-occuring diseases, a multilabel classification model was more appropriate. The labels were also encoded into binaries to help with generating a ROC/AUC curve to quantify the visualize the evaluation of the model after it has been trained.  The target labels were aggregated into a list of strings and stored in the `diseases_present` column to facilitate in multilabel classification. Although labels can be encoded and stored as a string of integers, it proved challenging to use a list of integers vs a list of string labels for the categorization hence the list of strings was used as the target variable instead.  Using the Keras `ImageDataGenerator()` and `flow_from_dataframe()` functions, the 


* Description of how data was split into training and testing sets 

The Preliminary Data is split into training, validation and testing groups. The samples were stratified on their labels to ensure that the distribution of disease labels remained consistent across different sets. The data sets were kept small: 300 training, 75 validation and 75 testing images to ensure efficient model training, focusing on actualization of the model instance prior to optimizing with a larger sample size. 

* Explanation of model choice, including limits and benefits

The model is based off of an online tutorial blog by [Vijayabhaskar J](https://vijayabhaskar96.medium.com/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24). The model has two iterations of a double convolutional layer with a maxpool layer that has relu activation functions. The machine will convolve across the image based on the dimensions specified and then the results will be pooled and sent to the next layer. The results are tensors which are just arrays of dot products. 

The next layer is the flatten layer that will reduce the dimensions of the tensors so that the regular neural neurons in the single dense hidden layer can process them. Then the data will be categorized in the output layer with a sigmoid function since this is a classification model. Relu functions througout the layers prevent the values from becoming having too much range and allows the model to process non-linear patters. 

Drop out layers are sandwiched between each major layer to ensure that when the model gets optimized, neurons that do not provide useful information are discarded to make the model more efficient.

Although there are other base models such as the [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) exist that can probably better classify the images efficiently without much computing power, the current model provides a framework to understand the basis of a simple convolutional neural network. Iterations of convolution and pooling layers leading to a flattened tensor that can be runthrough a traditional neural network to an output layer provides various points of modifications to optimize the model. Due to the size of the dataset, each training round can become memory intensive. Especially considering that the current model is not optimized to have sufficient dropout of neurons that can help with memory requirements. More epochs is necessary as the accuracy of the model during the training process was seen to be improving. Convolutional neural networks are ideal for image analysis especially when it comes to multilabel classification as there is very little need to pre-process the data. The Image data itself can be processed upon creating a generator instance that can modify the image accordingly to help the neural network digest the information.  


## Our Process & Communication Protocols

To facilitate collaboration, our group will primarily use our own Slack channel to share updates, questions and concerns. A series of checklists have been created for all team members to access via Google Sheets, 
allowing for quick, visual communications regarding what items still need to be completed.</br></br>




Create performance visualization for machine learning model and dashboard to view results.
