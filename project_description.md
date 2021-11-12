# Filling gaps in sparse spatio-temporal chlorophyll data in the Bering Sea
### Moses Lurbur (mlurbur), Max Pokutta (mpokutta), Leon Jiang (ljiang15)

## Introduction: 
- The National Oceanic and Atmospheric Administration (NOAA) closely monitors the health of the Bering Sea using a variety of biological metrics, one of which is chlorophyll levels. Chlorophyll is an indicator of Phytoplankton abundance, an important food source for juvenile fish and indication of overall ocean health.
- Chlorophyll levels are monitored via satellite, but the data contains large gaps due to high cloud cover in the Bering Sea.
- Our project will attempt to implement some form of structured prediction to fill the gaps in Chlorophyll data.
This topic by discussions about applications of DL with some of Moses’s past colleagues at NOAA. This project is an opportunity to apply DL to a highly relevant issue with potential impacts on fisheries regulations, policy and overall understanding of ocean health and dynamics in the Bering Sea.
- We will be consulting with a scientist at NOAA during the project to ensure we have a strong understanding of the problem, deliver meaningful results and have access to data.
- Concretely, we are aiming to solve a regression problem, predicting chlorophyll levels, under a supervised learning framework using historical data to train and test our model.

## Related Work:
A Spatiotemporal Prediction Framework for Air Pollution Based on Deep RNN
This paper compares a variety of techniques for predicting air pollution levels using a data set with missing values. They found that a deep recurrent neural net (LSTM) had the best performance. They detail ways to handle missing data values and work with spatio-temporal data.
Using Deep Learning to Fill Spatio-Temporal Data Gaps in Hydrological Monitoring Networks

## Data:
We are using data that has been shared by NOAA.
The data contains Chlorophyll, temperature, light, depth and ice data summarized over time and area intervals. The data is from 2003-present.
The data will require some preprocessing because we will have to create testing and training data while accounting for the sparse nature of the data.
 
Visualization of chlorophyll data from a single time period (8 days).

![Alt text](graph.jpg)
 
## Methodology: 
We are considering a few architectures, but, at a high-level, there are two main categories—one considers time (an RNN-like architecture) and the other does not (simple feed-forward).
Our base model will use just the latitude and longitude of the training data across the dataset’s time frame and attempt to predict the chlorophyll levels. This will likely be a simple feed-forward network (a few dense layers that outputs a single value) and may potentially incorporate a CNN that “learns” patterns of chlorophyll levels in the input for use in predicting its value.
Our sequential model (time-accounting) will either be an RNN-based architecture (LSTM/GRU) or, for better performance, a Transformer. The motivation is to capture some time-dependent information in the hidden state, like the movement of chlorophyll in the water, within the recent time frame, in order to predict the chlorophyll levels.
We will train our model using a testing set generated from our data. We have GCP credits and will likely use these to train our model on a virtual machine.
Metrics: 
Given that we are working on a unique problem with a form of data that is new to us (space and time variant data), we would consider the project a success if we are able to build a model that predicts chlorophyll values better than random.
Base: Train a model that uses spatial data to predict chlorophyll levels.
Target: Train a model that uses sparse spatio-temporal data to predict chlorophyll levels.
Stretch: Design a model that uses spatio-temporal data to predict chlorophyll levels, regardless of accuracy.
When evaluating our model, we will use an accuracy measurement based on how close the predicted value is to the actual value. We may also consider other measures of accuracy such as mean squared error.
We plan to perform a final evaluation of the trained model on an evaluation data set. 
 
## Ethics: 
- What broader societal issues are relevant to your chosen problem space? 
  - Our project has direct connections to environmental health and fisheries management. Chlorophyll levels are indicators of ocean health and having more accurate forecasting could help scientists and the public better understand how global warming is affecting our oceans.
- Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
  - If our model was used, mistakes could lead to changes in fishing legislation based on incorrect data. This would impact fisheries health and the fishermen whose livelihood is directly tied to the ocean and fishing regulations.
 
## Division of labor: 
- Moses
  - Data collection, cleaning
  - Building testing and training data sets
  - Final writeup 
- Max
  - Work on training and model architecture
  - Data preprocessing
  - Final writeup
- Leon
  - Work on testing and model architecture
  - Final writeup
