# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 23:38:53 2020

@author: Shaila Sarker
"""

import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding (taking inputs of ordinal data)
df = penguins.copy()
target = 'species' #output
encode = ['sex', 'island'] #inputs 

# encoding input columns ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]

# encoding output/target "species" 
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode) #applied the custom function named target_encode 

# Separating X and Y
X = df.drop('species', axis=1) #inputs all columns in df, except species
Y = df['species'] #output Y

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))