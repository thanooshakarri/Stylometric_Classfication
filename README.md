# Task

See https://uazhlt-ms-program.github.io/ling-582-course-blog/assignments/class-competition

Task URL: https://www.kaggle.com/c/ling-582-fa-2023-course-competition

Invitation URL: https://www.kaggle.com/t/f43bfaa8c4eff2644b60e83ef065c7e3


Given two spans of text, determine if both spans were produced by the same author.

This repository contains a trained model that classifies text, if both the texts are by the same author or not. My neural network is trained by using distil-RoBERTa embedding in a Bi-directional Gated Recurrent Unit.

Files:
```
├── README.md                <- Introduction of repository
├── model                    <- Trained model
├── requirements.txt         <- Python packages requirement file
├── data                     <- Dataset
|   |____ train.csv          <- Train data
|   |____ dev.csv            <- Validation data
|   |____ left-out-data.csv  <- left out data
|   |____ test.csv           <- Test data
|   |____ train-dev.csv      <- train+dev data
├── src                      <- Source code
|   |____ Classifier.py      <- Neural Network
|____ Main.ipynb             <- main function 
```
Usage:
```
Install the required libraries listed in requirements.txt by running pip install -r requirements.txt in your terminal.

Open the Main.ipynb notebook in Jupyter or any compatible platform.

The trained model is in the model folder.

The Classifier.py contains the files to train the model.

To make predictions upload the file namming it test.csv.

The test file should contain a column TEXT with the text.

The model is loaded and predicttions are made and the results are saved into predictions.csv file
```
