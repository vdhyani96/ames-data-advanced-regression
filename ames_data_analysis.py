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
# than the older houses. But the correlation is not that strong.

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

# e. average age of houses in different neighborhoods and average saleprices
plot_data = train[['Neighborhood', 'YearBuilt', 'SalePrice']]
# creating a new variable 'AgeOfHouse'
plot_data['AgeOfHouse'] = 2019 - plot_data['YearBuilt']
plot_data = plot_data.groupby('Neighborhood')['AgeOfHouse', 'SalePrice'].mean()

# plot with twin axis same as before
ax = sns.lineplot(x=plot_data.index, y='AgeOfHouse', data=plot_data, color='magenta', label='AgeOfHouse')
plt.xticks(rotation=90)
ax.legend(loc='upper left')
ax2 = plt.twinx()
ax2.legend(loc=0)
sns.lineplot(x=plot_data.index, y='SalePrice', data=plot_data, color='black', label='SalePrice')
# It's obvious from the plot that older houses are selling for low prices while newer 
# houses are selling for higher prices. Neighborhoods like OldTown,IDOTRR, SWISU having
# older houses on average are fetching less price for the houses. While neighborhoods with
# newer houses like NridgHt, NoRidge, StoneBr are getting higher prices for their
# houses on average
# checking the correlation of AgeOfHouse with the SalePrice
age_cor = plot_data.corr()
# -0.624 - a strong enough negative correlation

datase = train[['YearBuilt', 'SalePrice']]
temp_cor = datase.corr()

## 3. Writing a function to capture the desirability of the houses
# I can compare the output of the function with the SalePrice of the house as it is an
# accurate metric of a house's desirability
# I can use some variables that correlated well with SalePrice and can create a function
# using those. I can assign weights and can form a linear function. After that, can
# normalize the values
# Some variables:
# GrLivArea (0.708624)
# GarageArea (0.623431)
# TotalBsmtSF (0.613581)
# 1stFlrSF (0.605852)
# AgeOfHouse (-0.624)

desire = train[['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt']]
desire['AgeOfHouse'] = 2019 - desire['YearBuilt']
desire = desire.drop(columns='YearBuilt')
# normalizing the columns
d_norml = (desire - desire.min())/(desire.max() - desire.min())
# desirability index: creating a linear function of these variables according to their
# correlation values
d_norml['d_index'] = (np.exp(0.7086)*d_norml['GrLivArea'] + 
                        np.exp(0.6234)*d_norml['GarageArea'] +
                        np.exp(0.6136)*d_norml['TotalBsmtSF'] +
                        np.exp(0.6058)*d_norml['1stFlrSF'] - 
                        np.exp(0.624)*d_norml['AgeOfHouse'])

# normalize the d_index
d_norml['d_index'] = ((d_norml['d_index'] - d_norml['d_index'].min())/
                       (d_norml['d_index'].max() - d_norml['d_index'].min()))

d_norml['SalePrice'] = train['SalePrice']
# check correlation between the d_index and SalePrice
d_norml[['d_index', 'SalePrice']].corr()
# 0.80345 - This is even higher than the correlation of all of the individual features!

# Now, searching for ten most desirable and ten least desirable houses
# first plug these two columns into the main dataframe
dhouses = train
dhouses['d_index'] = d_norml['d_index']

# now sorting according to the desirability index
dhouses = dhouses.sort_values(by=['d_index'], ascending=False)

# 10 most desirable houses
print(dhouses.head(10))

# 10 least desirable houses
print(dhouses.tail(10))


## 4. Pairwise Distance Function - measuring the similarity of two properties
# Similar properties can be thought of having similar prices. Since, the desirability
# index is highly correlated to the SalePrice, I can take the difference between any
# two indices to find the similarity between two houses. 
# The Distance metric should be close to 0 for similar houses and
# large values for dissimilar houses. To magnify that range, I can use an exponential
# function as well
# To test, I can pick 3 houses and find the similarity between two pairs
house0 = d_norml['d_index'][0]
house1 = d_norml['d_index'][1]
house3 = d_norml['d_index'][3]
distance01 = 100**(abs(house0 - house1))
distance13 = 100**(abs(house1 - house3))
print(distance01)
print(distance13)
# From above, we can see that the distance between house1 and house3 is greater than
# the distance between house0 and house1. We can verify it using the SalePrice difference
# as well
Price0 = d_norml['SalePrice'][0]
Price1 = d_norml['SalePrice'][1]
Price3 = d_norml['SalePrice'][3]
Price01 = abs(Price0 - Price1)
Price13 = abs(Price1 - Price3)
print(Price01)
print(Price13)
# We see that Price difference between house1 and house3 is also greater than the price
# difference between house0 and house1.
# Another test with different set of houses:
house196 = d_norml['d_index'][196]
house990 = d_norml['d_index'][990]
house165 = d_norml['d_index'][165]
distance196_990 = 100**(abs(house196 - house990))
distance196_165 = 100**(abs(house196 - house165))
print(distance196_990)
print(distance196_165)
# now price comparison
Price196 = d_norml['SalePrice'][196]
Price990 = d_norml['SalePrice'][990]
Price165 = d_norml['SalePrice'][165]
Price196_990 = abs(Price196 - Price990)
Price196_165 = abs(Price196 - Price165)
print(Price196_990)
print(Price196_165)
# As we can see from both the distance metric and SalePrice comparison, house196 and 
# house990 are very similar, as opposed to house196 and house165, which are much 
# different


# This distance metric here is devised based on the price of the house. The 
# desirability index used for the calculation of the distance metric is also based on 
# the price of the house and is highly correlated with it. If similarity of two houses 
# manifests in some other form - like structure of the house, number of rooms, etc - then 
# this distance metric might not work. 






