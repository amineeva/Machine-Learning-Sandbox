# Import libraries and dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.model_selection import train_test_split #splits arrays/matrices into random train and test subsets
from sklearn.preprocessing import MinMaxScaler #scales each feature to a given range, same as using a z-score
from sklearn import metrics #scores functions, model metrics
from sklearn.svm import SVC #c-support vector classification| check Day 4 notes
from sklearn.linear_model import LogisticRegression #regularization applied by default

import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 



# User .describe() to get a description of the dataset
# To get a feel for the dataset create 2D and 3D plots of various features (scatter plots)
# Normalize your data using a z-score
# If the data is not already labeled with a numerical value, label it with a numerical value (1-10)
# Split data into training, validation, and test data sets
# Create the model object
# Train the model using the training data