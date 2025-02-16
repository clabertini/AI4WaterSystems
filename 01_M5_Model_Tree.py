# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:11:58 2025

@author: cbe001
"""
import numpy as np
import pandas as pd
import m5py
print(dir(m5py))
from m5py import M5Prime
from m5py.main import predict_from_leaves
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import networkx as nx
import matplotlib.pyplot as plt

'''0) LOAD THE DATA'''

# Define file path and check environment
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/My Drive/'
else:
    base_path = os.getcwd()  # Use current directory in Jupyter Notebook

# Ensure file exists before reading
file_name = 'Sieve-orig.xlsx'
file_path = os.path.join(base_path, file_name)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# First load the file with the data from Sieve
#file = 'Sieve-orig.xlsx'
#path = r'D:\07_IHE\05_TEACHING\02_WSD\2024-2025\M012-AI for Water Systems\DDM\MT/'
#df = pd.read_excel(path+file)

df = pd.read_excel(file_path)


'''1) SELECT THE INPUT FEATURES'''
# We need to arrange our dataset, so that each time step back of precipitation and 
# discharge is treated as a different variable.
# We first shift our rainfall data and copy them to separate columns. we take up to 5 steps back in time
for lag in range(1, 6):  # 10 steps back
    df[f'REt_lag{lag}'] = df['REt'].shift(lag)
    
# we do the same with discharge, taking up to 2 steps back in time
for lag in range(1, 3):  # 10 steps back
    df[f'Qt_lag{lag}'] = df['Qt'].shift(lag)
    
# we create the target (Qt+1) column
df['Qt+1'] = df['Qt'].shift(-1)

# now we remove the row with missing values (NaN = not a number)
df = df.dropna()
df.reset_index(inplace=True,drop=True)

# you can print the headers if you want to visualize your data set
df.head()

'''2) SPLIT THE DATASET BETWEEN TRAINING AND TESTING'''

# We split the data into training and testing, taking the first 300 rows for testing,
# and the rows from 301 onwards for training.
df_test = df.loc[:299]
df_train = df.loc[300:]

# Now we prepare the inputs (X) and the target (Y), being sure that X does not contain
# the target (Qt+1) and that Y contains only the target (Qt+1)

X_train = df_train.copy().drop(['Date','Qt+1'],axis=1).to_numpy()
X_test = df_test.copy().drop(['Date','Qt+1'],axis=1).to_numpy()

y_train = df_train['Qt+1'].to_numpy()
y_test = df_test['Qt+1'].to_numpy()

'''3) TRAIN THE M5 MODEL ON THE TRAINING SET'''
# Let's start with an unpruned model
m5_unpruned = M5Prime(use_pruning=False) # call the model
m5_unpruned.fit(X_train, y_train) # train the model

# Make predictions on the test set
y_unpruned = m5_unpruned.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_unpruned)
print(f'Mean Squared Error: {mse:.2f} (m³/s)')

# we can print the trees
regr_1_label = 'unpruned'
print("\n----- %s" % regr_1_label)
print(m5_unpruned.as_pretty_text())

# Let's now repeat with a pruned model
m5_pruned = M5Prime(use_pruning=True) # call the model
m5_pruned.fit(X_train, y_train) # train the model

# Make predictions on the test set
y_pruned = m5_pruned.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pruned)
print(f'Mean Squared Error: {mse:.2f} (m³/s)')
# we can print the trees
regr_2_label = 'pruned'
print("\n----- %s" % regr_2_label)
print(m5_pruned.as_pretty_text())



# Alternatively, you can change other parameters like the minimum number of 
# leaves in a node (default is 2) or the maximum tree depth (the default is None).
#Examples on how to use them are below:
model = M5Prime(min_samples_leaf=4,max_depth=8)
# note that in the line above you have just called the model, you have not fitted it 
# on the training data yet. Remember to call model.fit(X_train, y_train) to train the model.
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f} (m³/s)')
# you can always use these options with the pruned options as well, just add it in the parenthesis
# there is no difference in the ordering. Example: model = M5Prime(min_samples_leaf=4,max_depth=8,use_pruning=False)







