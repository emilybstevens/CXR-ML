# Examining Chest X-Rays Through Machine Learning
   
 
## Table of Contents
1. [Our Topic](https://github.com/emilybstevens/CXR-ML#our-topic)
2. [Project Outline](https://github.com/emilybstevens/CXR-ML#project-outline)
3. [Analysis](https://github.com/emilybstevens/CXR-ML#analysis)
4. [Our Data](https://github.com/emilybstevens/CXR-ML#our-data)
5. [Our Questions](https://github.com/emilybstevens/CXR-ML#our-questions)
6. [Google Slides](https://github.com/emilybstevens/CXR-ML#google-slides)
7. [Machine Learning Model](https://github.com/emilybstevens/CXR-ML#machine-learning-model)
8. [Dashboard](https://github.com/emilybstevens/CXR-ML#dashboard)
## Our Topic

### Overview
 
To create a Machine Learning model that can analyze a patient's chest X-rays to 
classify lung disease and flag X-rays displaying conditions with higher mortality rates in order to assist 
healthcare professionals in triaging high-risk patients faster. 
 
### Reasons for Selecting Topic

In recent years, the high rates of emergent lung disease have revealed a weakness in emergency medicine: emergency radiology. 
</br></br>
Traditionally, when a patient is admitted to the ER and has a chest X-ray performed (CXR), that CXR is entered into a waiting list to be read and interpreted by a radiologist. 
This interpretation is required for patients to receive any sort of active treatment or further hospitalization. 
From the time a CXR is taken, it can take several hours for the radiologist to return an interpretation to the ER staff, and without interpretation, ER physicians cannot recommend any type of treatment for their patient. 
This means that while the ER physician is waiting for the radiologist's report, the patient is effectively on standby, regardless of the severity of their symptoms.
</br></br>
Unfortunately, most hospitals do not have radiology triage systems in place. As a result, emergent patients requiring immediate surgery end up in the same waiting list as less dire patients, with no indicator 
way for radiologists to know which patient charts need to be prioritized. 
This lack of a useful CXR triage system results in longer healthcare wait times for emergent conditions requiring surgery, which can increase the mortality rate for those conditions. 
</br></br>
The goal of this model will be to create a machine learning model that can function as a triage system for this waiting list, flagging the higher-risk CXRs based on the presence of 
visual anomalies associated with higher-mortality conditions. With this triage system, radiologists can ensure that they are prioritizing the patients with potentially deadly conditions, thus 
reducing the rates of Adverse, Sentinel, and Never Events in ER settings by decreasing the amount of time from ER admittance to hospital admittance for emergent patients. 

## Project Outline
The scope of this project will be to create a prospective algorithm that can be marketed for use by ER staff for quick triage of CXRs, allowing emergent patients to be prioritized over non-emergent patients. 
A dashboard will be created that will include a visualization of what the application may look like once built, an overview of the algorithm's current performance and accuracy, an overview of the data set 
used to train and test the algorithm, an overview of why the algorithm was created, and section detailing information about each team member. 

## Analysis
A machine learning model will undergo supervised training to understand how to identify various CXR diagnoses. The outputs of the trained model will be analyzed for precision and accuracy, as well as improvement over the original, provisional model. 
</br></br>
In addition, the testing/training data set will be analyzed for trends based on various factors, including gender, view position, age, disease level, and triage level. 
Dashboards will be created for each of these variables, and will be viewable within the overall site created for this project. 
Tableau will be used to initially investigate potential trends in the data set, and once the team has narrowed down which visuals to include, Javascript & Plotly will be used to visualize the data within the created site. 

## Our Data 

### Data Source:
Data for this analysis will be sourced from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data?select=README_CHESTXRAY.pdf), utilizing chest X-rays compiled by the National Institutes of Health (NIH). 
The dataset includes 112,120 X-ray images from 30,805 unique patients. </br></br>
**Citation**: *Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
IEEE CVPR 2017, ChestX-ray8Hospital-ScaleChestCVPR2017_paper.pdf*

### Data Configuration:
The NIH Data_Entry_2007.csv and Mortality_Rate_Classification.csv files are loaded into the "Google Colab to AWS.ipynb" notebook where it is formatted and coded to Amazon Web Services (AWS). PGAdmin accesses the data files via the AWS connection and the queries from the "DB_csvfile_config" are run to split the multi-label x-ray images. Once the multi-labels are split apart, they are ranked against the mortality table to identify the triage level for each x-ray images. The "data_prep - pgadmin connection.ipynb" jupyter notebook accesses the data via PGAdmin and pre-processing the data for the machine learning model and training and testing datasets are created. From here the data_prep.ipynb is fed into the beta_net.ipynb file for the machine learning model. 

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

### Decsription of Data Preprocessing

The dataset included mroe than 100,000 images with 14 total unique disease labels. Each image could have multiple disease labels, with a maximum of 8 co-occuring labels within one image. The classification model seeks to identify these disease labels individually across all images. However, the distribution of each disease label is extremely unbalanced with images without a disease label `['No Finding']` representing more than half of the total images. Due to the large size of the dataset, it was relatively unbiased. Despite this, the number of images without disease labels required pruning to ensure that the training set included sufficient number of samples per disease label. 25,000 images without a disease label (1/3 the total number of disease free images) were extracted and divided between the different sets. Additionally, due to the small number of image samples for Hernia (<200), images with this disease labels were dropped from the set. This reduces the final number of images to around 80,000 for splitting between sets. Below is a bar graph representing the distribution of labels in the final dataset: 

![label_distribution](https://user-images.githubusercontent.com/99558296/181439630-2cb06127-8b15-4a34-b7f0-50921e1f4db7.png)

The target number of images in each of the sets were: 30,000 training images, 10,000 validation images, and 10,000 testing images. These sets were separated in order to ensure that the model does not encounter the same images between different sets. The sets were split using scikit-learn's `train_test_split()` function by stratifying the disease labels to ensure that the distribution of the labels were preserved across the different sets.

### Description of feature engineering and feature selection, including their decision-making process

Since the images were spread across 12 total folders with around 10,000 images per folder, the paths towards the individual images were saved in a new column of the dataframe. This `['path']` column was used as the x variable for the neural network. The labels located in the `['Finding Labels']` column were one-hot encoded using scikit-multilearn's `MultiLabelBinarizer()` to produce 13 unique columns of disease labels containing the binarized target variables. Images without a disease label were 0 across all 13 columns. The final disease labels were: 

- Atelectasis,
- Cardiomegaly,
- Consolidation,
- Edema,
- Effusion,
- Emphysema,
- Fibrosis,
- Infiltration,
- Mass,
- Nodule,
- Pleural_Thickening,
- Pneumonia,
- Pneumothorax

Finally, each datasets was split between the `X = ['path']` and `y = ['disease_labels']` before they were fed into custom image-pre-processing and batching functions that generate batches of image-data. Each dataset produces a batch of 32 images with their associated labels for the training and validation sets and without labels for the testing set as this prevents clogging in the processing memory of the machine. This is especially important for the training set as random image augmentations performed in the set were saved in cache. The custom batching function uses tensorflow's `autotune` class to adapt the batching based on the memory of the system. 

The image augmentation was lar


Since each sample could have up to 8 co-occuring diseases, a multilabel classification model was more appropriate. The labels were encoded into binaries to help with generating a ROC/AUC curve to quantify the visualize evaluation of the model after being trained.  The target labels were aggregated into a list of strings and stored in the `diseases_present` column to facilitate in multilabel classification. Although labels can be encoded and stored as a string of integers, it proved challenging to use a list of integers vs a list of string labels for the categorization hence the list of strings was used as the target variable instead.  Using the Keras `ImageDataGenerator()` and `flow_from_dataframe()` functions, we are able to perform image augmentation while training the model. Pandas dataframe and a path directory are taken in to generate batches of augmented/normalized data.


### Description of how data was split into training and testing sets 

The Preliminary Data is split into training, validation and testing groups. The samples were stratified on their labels to ensure that the distribution of disease labels remained consistent across different sets. The data sets were kept small: 300 training, 75 validation and 75 testing images to ensure efficient model training, focusing on actualization of the model instance prior to optimizing with a larger sample size. 

### Explanation of model choice, including limits and benefits

The model is based off of an online tutorial blog by [Vijayabhaskar J](https://vijayabhaskar96.medium.com/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24). The model has two iterations of a double convolutional layer with a maxpool layer that has relu activation functions. The machine will convolve across the image based on the dimensions specified and then the results will be pooled and sent to the next layer. The results are tensors which are just arrays of dot products. 

The next layer is the flatten layer that will reduce the dimensions of the tensors so that the regular neural neurons in the single dense hidden layer can process them. Then the data will be categorized in the output layer with a sigmoid function since this is a classification model. Relu functions througout the layers prevent the values from having too much range and allows the model to process non-linear patterns. 

Drop out layers are sandwiched between each major layer to ensure when the model gets optimized, neurons that do not provide useful information are discarded to make the model more efficient.

Although there are other base models that may better classify the images efficiently without much computing power, such as the [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html), the current model provides a framework to understand the basis of a simple convolutional neural network. Iterations of convolution and pooling layers leading to a flattened tensor can be runthrough a traditional neural network to an output layer which provides various points of modifications to optimize the model. Due to the dataset size, each training round can be memory intensive especially considering the current model is not optimized to have sufficient dropout of neurons to help with memory requirements. During the training process accuracy was seen improving, therefore more epochs are necessary for the accuracy of the model. Convolutional neural networks are ideal for image analysis especially when it comes to multilabel classification as there is very little need to pre-process the data. The Image data itself can be processed upon creating a generator instance that can modify the image accordingly to help the neural network digest the information.  

### Explanation of Changes in Model Choice
No changes in model choice were made. 
### Description of Model Training, Current and Future
The model has currently been trained on a maximum of 10 epochs. Future training will involve increasing to 50 epochs, and redefining the way images are processed. 
### Description of Current Accuracy Score
Accuracy for the machine changes depend on the category. Some categories have a higher AUC-ROC score, meaning they are more accurate. The highest achieved ROC score was 0.88 for fibrosis. 

## Dashboard

To view preliminary dashboard, please click [here](https://emilybstevens.github.io/xraydar/). </br></br>  
Please note: The data dashboard is still a work in progress. The intent is to eventually have two separate pages: 
the first page will contain pre-fabricated Tableau spreadsheets to interact with, 
while the other page will include two separate Javascript dashboards (one dedicated to inidividual sample data, and one dedicated to filtering data by various demographics). </br></br> 
The Javascript page is currently located under `Data->Interactive Dashboard`. Please note, the code is currently running very slowly, as there are over 100k+ datapoints being accessed. 
The current dash requires performance enhancement for speed. Give it time to load, or it will crash. </br></br> 
The secondary Javascript dash is currently in the works. </br></br> 
The Javascript page is currently located under `Data->Overview`.</br></br> 
Images from initial analysis, as well as machine learning data can found under the `Performance` tab. 


