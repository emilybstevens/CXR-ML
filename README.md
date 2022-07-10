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
* Description of preliminary feature engineering and preliminary feature selection, including their decision-making process
* Description of how data was split into training and testing sets
* Explanation of model choice, including limits and benefits

## Our Process & Communication Protocols

To facilitate collaboration, our group will primarily use our own Slack channel to share updates, questions and concerns. A series of checklists have been created for all team members to access via Google Sheets, 
allowing for quick, visual communications regarding what items still need to be completed.</br></br>




Create performance visualization for machine learning model and dashboard to view results.
