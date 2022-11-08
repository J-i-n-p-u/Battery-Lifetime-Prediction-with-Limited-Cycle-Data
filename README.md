# Battery-Lifetime-Prediction-with-Limited-Cycle-Data
Applied machine-learning models, including linear regression, random forest regression, convolutional neural networks, and recurrent neural networks to make predictions on cell life. 

<img src='battery.png'/>


## Description
Accurately predicting the remaining useful lifetime of batteries is critical for accelerating technological development and creating a paradigm shift in battery usage. Data-driven approaches,based on large datasets, provide a physical-model agnostic way to predict the health status of batteries with high accuracy. However, most datadriven methods on battery life prediction often rely on features extracted from a hundred cycles worth of data for a given cell, making it computationally inefficient and incompatible with on-board application.

The course project (CS 329P Practical Machine Learning) applied machine-learning models, including linear regression, random forest regression, convolutional neural networks, and recurrent neural networks to make predictions on cell life. Our best model achieve a 7.5% prediction error given the data of only 5 cycles. ([Report](https://github.com/J-i-n-p-u/Battery-Lifetime-Prediction-with-Limited-Cycle-Data/blob/main/Battery%20Lifetime%20Prediction%20with%20Limited%20Cycle%20Data.pdf), [Slides](https://github.com/J-i-n-p-u/Battery-Lifetime-Prediction-with-Limited-Cycle-Data/blob/main/Battery%20Lifetime%20Prediction.pdf), [Video](https://github.com/J-i-n-p-u/Battery-Lifetime-Prediction-with-Limited-Cycle-Data/blob/main/Battery%20Lifetime%20Prediction.mp4))

**Key Words: Battery Lifetime Prediction, CNN, Bi-LSTM, Confidence Interval**

## Data Preprocess
The dataset preprocessing method refers to the paper [Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41560-019-0356-8). The preprocessed dataset file (`processed_data.pkl`) in our code is generated from this [script](https://github.com/dsr-18/long-live-the-battery). The raw dataset can be found [here](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).

## Installation
The project uses Python3.9, Tensorflow 2.5 - GPU
