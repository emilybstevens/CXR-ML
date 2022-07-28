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

![label_distribution](https://user-images.githubusercontent.com/99558296/181464938-7966472f-e658-475c-8b44-7e07010e8d98.png)

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

Finally, each datasets was split between the `X = ['path']` and `y = ['disease_labels']` before they were fed into custom image-pre-processing and batching functions that generate batches of image-data. The custom pre-processing occurs using `parse_function()` and the batching occurs using `create_dataset()`. Each dataset produces a batch of 32 images with their associated labels for the training and validation sets and without labels for the testing set as this prevents clogging in the processing memory of the machine. This is especially important for the training set as random image augmentations performed in the set were saved in cache. The custom batching function uses tensorflow's `autotune` class to adapt the batching based on the memory of the system. 

The base feature extraction layer and the image augmentations were based on the literature by [Giełczyk et al](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949). The literature points to a ResNet pre-trained feature extraction layer: [ResNet101](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) to achieve better classification results. Additionally, image augmentation through [histogram equalization](https://www.tensorflow.org/addons/api_docs/python/tfa/image/equalize) and [guassian blurring](https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d) were deemed as the ideal agumentation methods and were performed using tensorflow-addons module. After, the image tensors were reshaped from (1024 x 1024 x 3) to (224, 224, 3) and normalized between 0-1 to accomodate the input specifications of ResNet101. However, to truly discourage overfitting of data, training images were subjected to a [preprocessing `Sequential()` layer](https://keras.io/guides/preprocessing_layers/) that is housed within the batching function, external to the CNN. The training set was also shuffled in batches of 256 images. The preprocessing Sequential layer layer included RandomFlip horizontally, RandomZoom and RandomRotation operations that augment the training data to discourage overfitting: 
```
trainAug = Sequential([
	RandomFlip("horizontal"),
	RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	RandomRotation(0.05)
])
```

### Description of how data was split into training and testing sets 

The target number of images in each of the sets were: 30,000 training images, 10,000 validation images, and 10,000 testing images. These sets were separated in order to ensure that the model does not encounter the same images between different sets. The sets were split using scikit-learn's `train_test_split()` by stratifying the disease labels to ensure that the distribution of the labels were preserved across the different sets.

### Explanation of model choice, including limits and benefits

Using the literature previously discussed, the model was built with a base ResNet101 feature extraction layer that was fine-tuned (re-trained) using transfer-learning. This layer extracts the features using iterations of convolution and maxpooling layers supplemented with ReLU activation functions. The feature extraction layer was followed by a global average pooling 2D layer that pools and further distills the tensor outputs from the extraction layer. This is further distilled by a flatten layer that reduces the dimensionality of the tensors to reduce processing requirements. Dropout layers encapsulate the fully connected neuron layer that enables backpropagation during training to ensure that the features are properly processed without being overfitted. Finally, the output layer includes 13 neurons with a sigmoid activation function to calculate the individual probabilities of 13 disease labels. The model was compiled using the Adam optimizer with a learning rate of (1e-4) and `binary_crossentropy` was used as the loss function to ensure proper training of a multilabel classifier. The metrics monitored were `binary_accuracy` and `mae` to assess the model's accuracy in indentifying the true labels. The model was fit using a callback function that saves the weights every 50 steps in addition to early stopping the training if the `val_loss` metric does not improve by 0.001 within 2 epochs. Below is a summary of the model: 

<img width="269" alt="d_model_summary" src="https://user-images.githubusercontent.com/99558296/181454566-eee20557-0292-4f9f-a69d-112d0fbd33c2.png">

The model is designed to identify patterns associated with each disease label within the images as such proper image augmentation is necessary for the feature extraction layer to accurately identify the disease labels. However, the model is limited to reciving images in batches and not single images as ResNet101 demands image tensor shapes to be (num_images, 224, 224, 3). This makes it ideal for its intended use of being a high-throughput image classifier to aid in patient triage. Furthermore, ResNet101's requirement of images across 3 channels (RGB) impacts the speed and efficiency of the image classification since RGB (224, 224, 3) images reduced to greyscale (224, 224, 1) offers lower memory cost and lower noise.

### Explanation of Changes in Model Choice

Initially, the model was trained on a CPU but due to the large datasets and heavy memory requirements, the training was later performed on google colab using google's backend TPU. This dramatically reduced the training time and procesing power required. Although the initial alpha model utilized [MobileNetV2](https://keras.io/api/applications/mobilenet/) due to its lightweight architecture that is suitable for CPU systems, ResNet101 was ultimiately chosen as the feature extraction layer based on its higher F1 score according to the [literature](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949) previously discussed. The goal of the model is to correctly identify the presence of multiple diseases in order to evaluate the triage level of the patient associated with the image. Thus, the F1 score is crucial as it properly evaluates the model by only focusing on the True Positives. Despite this, the highest final F1 score was 0.325 from the delta_model. This is in addition to the fine tuning of the feature extraction layer. Among the myriad of reasons for this low F1 score, the most likely culprit is the noise from images without disease labels as it prevents the model from receiving enough input images across images with disease labels that have a lower distribution relative the sample size. The distribution of labels within the training set along with delta_model's confusion matrices are displayed below for reference. **Note labels with low true positives correlate with lower image counts in the training set**: 

![train_dist](https://user-images.githubusercontent.com/99558296/181465008-76873b54-5d28-4d57-a9eb-847299341b39.png)

![delta_cm_Atelectasis](https://user-images.githubusercontent.com/99558296/181460764-c3bab1a9-4274-472b-a225-6fb8f1be9d16.png)
![delta_cm_Cardiomegaly](https://user-images.githubusercontent.com/99558296/181460782-89615623-3ae8-45c4-bdea-5add6bdaf8b4.png)
![delta_cm_Consolidation](https://user-images.githubusercontent.com/99558296/181460797-cc69cf03-8486-4a86-8288-f158453e5357.png)
![delta_cm_Edema](https://user-images.githubusercontent.com/99558296/181460803-ecea2931-fc71-4645-ab97-9a0e96ffcb5b.png)
![delta_cm_Effusion](https://user-images.githubusercontent.com/99558296/181460816-fbd6404e-d0cf-470d-98be-d734670d4d26.png)
![delta_cm_Emphysema](https://user-images.githubusercontent.com/99558296/181460836-ddae772f-f5a1-4266-a92b-a912e4f9a65c.png)
![delta_cm_Fibrosis](https://user-images.githubusercontent.com/99558296/181460845-27d17dbf-a3e8-4786-baf5-0752544ab8cb.png)
![delta_cm_Infiltration](https://user-images.githubusercontent.com/99558296/181460855-b5a0fd32-ba6e-4243-b12c-f449632e8894.png)
![delta_cm_Mass](https://user-images.githubusercontent.com/99558296/181460860-aad471de-c5ed-4f63-a37e-a446252cb34a.png)
![delta_cm_Nodule](https://user-images.githubusercontent.com/99558296/181460865-b01a734d-a0f1-43eb-b60b-412082cc1472.png)
![delta_cm_Pleural_Thickening](https://user-images.githubusercontent.com/99558296/181460891-797fd815-b69c-4d7e-a0d0-2f503a6f3f26.png)
![delta_cm_Pneumonia](https://user-images.githubusercontent.com/99558296/181460903-0f5f0d00-d7d0-445a-8e36-82e05c4ab237.png)
![delta_cm_Pneumothorax](https://user-images.githubusercontent.com/99558296/181460915-67ccc170-1f8d-4a84-a4fb-31fb4df9751f.png)

### Description of Model Training, Current and Future
The current delta_model has been trained to 6 epochs. Although the architecture can be further fine tuned, the dataset itself should be reprocessed to produce a better distribution of image samples across all the disease labels. This could mean: 
- reducing the total number of images for the training set. 
- reducing the noise through a reduction of images without disease labels. 
- reducing the number of images from disease labels with higher image counts if overfitting is suspected. 
Once the dataset is optimized, more base pre-trained layers should be explored to see if there is a better model that is suited for Multilabel classification. Ideally, this model should be able to process images in greyscale without batching required. This would further reduce noise and alleviate memory and processing requirements for the model to be viable across all processing systems. 

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


