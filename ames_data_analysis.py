# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:54:27 2019

@author: DHYANI
"""

import numpy as np
import pandas as pd
import seaborn
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


