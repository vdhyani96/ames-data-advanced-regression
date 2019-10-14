# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:54:27 2019

@author: DHYANI
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt

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
sns.heatmap(corr, cmap=cmap)

## 1. Capturing most positive and most negative pairwise correlations
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


## 2. Some informative plots for this dataset
# a. Scatter plot between YearBuilt and SalePrice
sns.lmplot(data = train[['YearBuilt', 'SalePrice']], x = 'YearBuilt', y = 'SalePrice')
# the scatter plot and the trendline show that newer houses are sold for higher prices 
# than the older houses. But the correlation is not that large.

# b. boxplot of general zoning classification vs SalePrice
plot_data = train[['MSZoning', 'SalePrice']]
plot_data['MSZoning'].loc[plot_data['MSZoning'] == 'RL'] = 'Residential Low Density'
plot_data['MSZoning'].loc[plot_data['MSZoning'] == 'RM'] = 'Residential Medium Density'
plot_data['MSZoning'].loc[plot_data['MSZoning'] == 'C (all)'] = 'Commercial'
plot_data['MSZoning'].loc[plot_data['MSZoning'] == 'FV'] = 'Floating Village Residential'
plot_data['MSZoning'].loc[plot_data['MSZoning'] == 'RH'] = 'Residential High Density'
ax = sns.boxplot(x='MSZoning', y='SalePrice', data=plot_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

# c. bargraph of LotShape and in the same the average saleprice
plot_data = train[['LotShape', 'SalePrice']]
plot_data['LotShape'].loc[plot_data['LotShape'] == 'Reg'] = 'Regular'
plot_data['LotShape'].loc[plot_data['LotShape'] == 'IR1'] = 'Slightly irregular'
plot_data['LotShape'].loc[plot_data['LotShape'] == 'IR2'] = 'Moderately Irregular'
plot_data['LotShape'].loc[plot_data['LotShape'] == 'IR3'] = 'Irregular'
sns.barplot(x='LotShape', y='SalePrice', data=plot_data)
plt.xticks(rotation=30)

# Surprisingly, the average SalePrice of the irregular properties are higher than the
# regular properties. Median SalePrice values also show similar results.
sns.barplot(x='LotShape', y='SalePrice', data=plot_data, estimator=np.median)
plt.xticks(rotation=30)

# d. neighborhoods with overall condition of the house and average saleprice
# get the average condition of houses in neighborhoods
plot_data = train[['Neighborhood', 'OverallCond', 'SalePrice']]
plot_data = plot_data.groupby('Neighborhood')['OverallCond', 'SalePrice'].mean().sort_values(
        by='OverallCond')

# twin plots to see how overall condition and average saleprice of the houses
# vary in each neighborhood
ax = sns.lineplot(x=plot_data.index, y='OverallCond', data=plot_data, color='r', label='OverallCond')
plt.xticks(rotation=90)
ax.legend(loc='upper left')
ax2 = plt.twinx()
ax2.legend(loc=0)
sns.lineplot(x=plot_data.index, y='SalePrice', data=plot_data, color='b', ax=ax2, label='SalePrice')
# There are some surprising results: Some neighborhoods having the least value of
# overall condition of houses are having the highest SalePrices. (NoRidge, Nridght,
# StoneBrook). While, a few others like OldTown, BrkSide, and Crawfor have good 
# house conditions on average, but the SalePrice is low.

# e. 





