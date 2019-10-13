# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:54:27 2019

@author: DHYANI
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os

os.chdir('C:\\Users\\admin\\Desktop\\PostG\\GRE\\Second Take\\Applications\\Univs\\Stony Brook\\Fall 19 Courses\\DSF\\Homework 3\\house-prices-advanced-regression-techniques')


# load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# selecting few "interesting" variables for pairwise correlation analysis
sub = ['SalePrice', 'LotFrontage', 'LotArea', 'OverallCond', 'YearRemodAdd', 'MasVnrArea', 
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd',
       'GarageArea', 'YrSold']

cor_data = train[sub]
corr = cor_data.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
seaborn.heatmap(corr, cmap=cmap)

# Capturing most positive and most negative pairwise correlations
most_positive = []
most_negative = []
for col in corr.columns:
    sorted = corr[col].sort_values()
    max_val = sorted[-2]
    max_row = sorted.index[-2]
    most_positive.append((col, max_row, max_val))
    min_val = sorted[0]
    min_row = sorted.index[0]
    most_negative.append((col, min_row, min_val))
    


# Following tables store the pairwise most positive and negative correlations
print(np.array(most_positive).reshape(14, 3))
print(np.array(most_negative).reshape(14, 3))    

# GrLivArea (Above grade (ground) living area square feet) has the highest
# correlation with the SalePrice (target variable) of 0.7086
# Other notably high correlations are between 
# TotalBsmtSF (Total square feet of basement area) and 
# 1stFlrSF (First Floor square feet) of 0.8195, and between 
# GrLivArea (Above grade (ground) living area square feet) and 
# TotRmsAbvGrd (Total rooms above grade) of 0.8255

# Surprisingly, SalePrice has a very weak negative correlation with the
# OverallCond (overall condition of the house) of just -0.07785
# Other highly negative correlations are between
# 1stFlrSF (First Floor square feet) and 2ndFlrSF (Second floor square feet) 
# of -0.2026, and between TotalBsmtSF (Total square feet of basement area) 
# and 2ndFlrSF (Second floor square feet) of -0.1745

# Some other highly correlated variables with the target variable (SalePrice) are:
# GarageArea (0.623431)
# TotalBsmtSF (0.613581)
# 1stFlrSF (0.605852)
# TotRmsAbvGrd (0.533723)
# YearRemodAdd (0.507101)


