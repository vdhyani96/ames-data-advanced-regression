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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import hdbscan


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
most_positive = np.array(most_positive).reshape(14, 3)
most_negative = np.array(most_negative).reshape(14, 3)
pos = pd.DataFrame({'Variable':most_positive[:,0], 'MaxCorrelation':most_positive[:,1], 'Correlation':most_positive[:,2]})
neg = pd.DataFrame({'Variable':most_negative[:,0], 'MinCorrelation':most_negative[:,1], 'Correlation':most_negative[:,2]})
pos = pos.sort_values('Correlation', ascending=False)
pos.reset_index(drop=True, inplace=True)
neg = neg.sort_values('Correlation')
neg.reset_index(drop=True, inplace=True)
print(pos)
print("\n")
print(neg)     

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
sns.lmplot(data = d_norml[['d_index', 'SalePrice']], x='d_index', y='SalePrice', scatter_kws={'alpha':0.3})
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

# now create a master table for average desirability score across all neighborhoods
master = pd.DataFrame({'Neighborhood':train['Neighborhood'], 'd_index':d_norml['d_index']})
master = master.groupby('Neighborhood')['d_index'].mean()

# desirability of neighborhoods based on average d_index
sns.barplot(master, master.index)
# in tabular form
print(master.sort_values(ascending=False))


## 4. Pairwise Distance Function - measuring the similarity of two properties
# Similar properties can be thought of having similar prices. Since, the desirability
# index is highly correlated to the SalePrice, I can take the difference between any
# two indices to find the similarity between two houses. 
# The Distance metric should be close to 0 for similar houses and
# large values for dissimilar houses. To magnify that range, I can use an exponential
# function as well
# To test, I can pick 3 houses and find the similarity between two pairs
# can also include normalized SalePrice
d_norml['SalePrice'] = ((d_norml['SalePrice'] - d_norml['SalePrice'].min())/
                        (d_norml['SalePrice'].max() - d_norml['SalePrice'].min()))
house0 = d_norml['d_index'][0] + d_norml['SalePrice'][0]
house1 = d_norml['d_index'][1] + d_norml['SalePrice'][1]
house3 = d_norml['d_index'][3] + d_norml['SalePrice'][3]
distance01 = 100**(abs(house0 - house1))
distance13 = 100**(abs(house1 - house3))
print(distance01)
print(distance13)
# From above, we can see that the distance between house1 and house3 is greater than
# the distance between house0 and house1. We can verify it using the SalePrice difference
# as well
Price0 = train['SalePrice'][0]
Price1 = train['SalePrice'][1]
Price3 = train['SalePrice'][3]
Price01 = abs(Price0 - Price1)
Price13 = abs(Price1 - Price3)
print(Price01)
print(Price13)
# We see that Price difference between house1 and house3 is also greater than the price
# difference between house0 and house1.
# Another test with different set of houses:
house196 = d_norml['d_index'][196] + d_norml['SalePrice'][196]
house990 = d_norml['d_index'][990] + d_norml['SalePrice'][990]
house165 = d_norml['d_index'][165] + d_norml['SalePrice'][165]
distance196_990 = 100**(abs(house196 - house990))
distance196_165 = 100**(abs(house196 - house165))
print(distance196_990)
print(distance196_165)
# now price comparison
Price196 = train['SalePrice'][196]
Price990 = train['SalePrice'][990]
Price165 = train['SalePrice'][165]
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


## 5. Clustering of houses using the distance metric

sns.lmplot(data = d_norml[['d_index', 'SalePrice']], x='d_index', y='SalePrice')
# Can we cluster the above points using the distance metric?
# Would need to create a (square) pairwise distance matrix first using my distance logic
# separate the d_index and SalePrice
clus = np.array(d_norml[['d_index', 'SalePrice']])
# Compute the pairwise distance matrix using my distance metric
distance_matrix = pairwise_distances(clus, 
                        metric=lambda x, y: 100**(abs((x[0]+x[1])-(y[0]+y[1]))))

# Now run the HDBSCAN clustering algorithm on the data points
clusterer = hdbscan.HDBSCAN(metric='precomputed')
clusterer.fit(distance_matrix)
clusterer.labels_
clusterer.labels_.max()
# Only three clusters were produced
print(np.unique(clusterer.labels_, return_counts=True))

# visualizing
color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*clus.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)

train['Neighborhood'][clusterer.labels_==0]
# This cluster has houses in Neighborhoods that are new and sell for higher prices
# mostly NridgeHt, NoRidge and StoneBr
train['Neighborhood'][clusterer.labels_==2]
# This cluster also has some specific Neighborhoods where houses are new and sell for
# high prices, like NoRidge, StoneBr, but it also has some other mix of Neighborhoods
train['Neighborhood'][clusterer.labels_==1]
# All the other Neighborhoods (apart from the noise) are in this cluster. This is the
# biggest cluster of all
train['Neighborhood'][clusterer.labels_==-1]
# This is the list of noise or outliers

# Can try any other clustering if I want - Kmeans
kmeans = KMeans(n_clusters=5, random_state=519).fit(distance_matrix)
print(np.unique(kmeans.labels_, return_counts=True))
# visualizing
new = np.append(clus, np.reshape(kmeans.labels_, (1460,1)), axis=1)
new = pd.DataFrame({'Col1':new[:, 0], 'Col2':new[:, 1], 'Cluster':new[:, 2]+1})
color_palette = sns.color_palette('deep', 5)
sns.scatterplot(data=new, x='Col1', y='Col2', hue='Cluster', legend='full', palette=color_palette)

# Let's check the distribution of neighborhoods in each clusters
# 1st cluster
var = np.unique(train['Neighborhood'][kmeans.labels_==0], return_counts=True)
var1 = np.array(var[0], dtype='str')
var2 = var[1]
var = pd.DataFrame({'Neighborhood': var1, 'Count': var2})

# join var with the master table
var = var.merge(master, left_on='Neighborhood', right_on=master.index, how='inner')
var['desirability'] = 'low'
var['desirability'][var['d_index']>0.2] = 'moderate'
var['desirability'][var['d_index']>0.3] = 'high'
var['desirability'][var['d_index']>0.4] = 'very high'
sns.barplot(data=var, y='Neighborhood', x='Count', hue='desirability')
# As we can see, this cluster consists of a mix of moderate, high and very highly desirable neighborhoods

# 2nd Cluster
var = np.unique(train['Neighborhood'][kmeans.labels_==1], return_counts=True)
var1 = np.array(var[0], dtype='str')
var2 = var[1]
var = pd.DataFrame({'Neighborhood': var1, 'Count': var2})

# join var with the master table
var = var.merge(master, left_on='Neighborhood', right_on=master.index, how='inner')
var['desirability'] = 'low'
var['desirability'][var['d_index']>0.2] = 'moderate'
var['desirability'][var['d_index']>0.3] = 'high'
var['desirability'][var['d_index']>0.4] = 'very high'

# This cluster has just a single Neighborhood NoRidge with very high desirability

# 3rd Cluster
var = np.unique(train['Neighborhood'][kmeans.labels_==2], return_counts=True)
var1 = np.array(var[0], dtype='str')
var2 = var[1]
var = pd.DataFrame({'Neighborhood': var1, 'Count': var2})

# join var with the master table
var = var.merge(master, left_on='Neighborhood', right_on=master.index, how='inner')
var['desirability'] = 'low'
var['desirability'][var['d_index']>0.2] = 'moderate'
var['desirability'][var['d_index']>0.3] = 'high'
var['desirability'][var['d_index']>0.4] = 'very high'
sns.barplot(data=var, y='Neighborhood', x='Count', hue='desirability')
# From the plot, it looks like this cluster contains mostly low and moderately desirable neighborhoods
# but most are moderate

# 4th Cluster
var = np.unique(train['Neighborhood'][kmeans.labels_==3], return_counts=True)
var1 = np.array(var[0], dtype='str')
var2 = var[1]
var = pd.DataFrame({'Neighborhood': var1, 'Count': var2})

# join var with the master table
var = var.merge(master, left_on='Neighborhood', right_on=master.index, how='inner')
var['desirability'] = 'low'
var['desirability'][var['d_index']>0.2] = 'moderate'
var['desirability'][var['d_index']>0.3] = 'high'
var['desirability'][var['d_index']>0.4] = 'very high'
sns.barplot(data=var, y='Neighborhood', x='Count', hue='desirability')
#plt.xticks(np.arange(min(var['Count']), max(var['Count'])+1, 1.0))
# This cluster also has low and moderately desirable neighborhoods, but most are low

# 5th cluster
var = np.unique(train['Neighborhood'][kmeans.labels_==4], return_counts=True)
var1 = np.array(var[0], dtype='str')
var2 = var[1]
var = pd.DataFrame({'Neighborhood': var1, 'Count': var2})

# join var with the master table
var = var.merge(master, left_on='Neighborhood', right_on=master.index, how='inner')
var['desirability'] = 'low'
var['desirability'][var['d_index']>0.2] = 'moderate'
var['desirability'][var['d_index']>0.3] = 'high'
var['desirability'][var['d_index']>0.4] = 'very high'
sns.barplot(data=var, y='Neighborhood', x='Count', hue='desirability')
plt.xticks(np.arange(min(var['Count']), max(var['Count'])+1, 1.0))
# This cluster has the most desirable neighborhoods of Ames - NoRidge, NridgHt, and StoneBr

# Tried with 9 clusters but it creates a mix of neighborhoods in each cluster that are not much compatible with the
# categories of desirability


## 6. Linear regression model to predict the SalePrice
# I will pick the features that are most correlated with the SalePrice
xlin = d_norml.iloc[:, :-1]
ylin = d_norml['SalePrice']
lreg = LinearRegression().fit(xlin, ylin)
lreg.score(xlin, ylin)
# The Rsquared value is 0.7132, which shows that about 71.3% variance in the SalePrice is explained
# by the predictors - not a bad value
pred = lreg.predict(xlin)
rmse = np.sqrt(mean_squared_error(ylin, pred))
print(rmse)
# rmse = 0.0591
# Since all the variables were normalized before training, I can simply compare the coefficients of
# the model to see the most important variable
lreg.coef_
# We can see that the most important variable here is GrLivArea (living area above ground) with
# coeff value = 0.4856



## 7. Integrate one external dataset to improve prediction
# 30-Year Fixed Rate Mortgage Average in the United States dataset
# Freddie Mac, 30-Year Fixed Rate Mortgage Average in the United States [MORTGAGE30US], retrieved from FRED, Federal Reserve Bank of St. Louis; 
# https://fred.stlouisfed.org/series/MORTGAGE30US, October 19, 2019.
# Copyright, 2016, Freddie Mac. Reprinted with permission.
# I will take this dataset and join it with the train dataset on the year of sale, in order to get
# the mortgage rate for that year
mgage_data = pd.read_csv('External_dataset_MORTGAGE30US.csv')
# have to separate the date column into year and month
mgage_data['Year'] = pd.to_datetime(mgage_data['DATE']).dt.year
mgage_data['Month'] = pd.to_datetime(mgage_data['DATE']).dt.month
# now join train with the mgage_data on Year and Month
enhanced = train.merge(mgage_data, left_on=['YrSold', 'MoSold'], right_on=['Year', 'Month'], how='inner')
enhanced = enhanced.drop(columns=['DATE', 'Year', 'Month'])
enhanced.rename(columns={'MORTGAGE30US':'Mortgage'}, inplace=True)
# check the correlation of mortgage with the SalePrice
sns.lmplot(data=enhanced, x='Mortgage', y='SalePrice')
# pretty bad, can try training the model (with normalizing Mortgage)
xlin2 = d_norml.iloc[:, :-1]
xlin2['Mortgage'] = enhanced['Mortgage']
# normalize Mortgage
xlin2['Mortgage'] = ((xlin2['Mortgage'] - xlin2['Mortgage'].min())/
                    (xlin2['Mortgage'].max() - xlin2['Mortgage'].min()))
ylin2 = d_norml['SalePrice']
lreg2 = LinearRegression().fit(xlin2, ylin2)
lreg2.score(xlin2, ylin2)
# The Rsquared value is 0.7134, which is nearly the same as with the model without Mortgage variable
pred2 = lreg2.predict(xlin2)
rmse2 = np.sqrt(mean_squared_error(ylin2, pred2))
print(rmse2)
# rmse = 0.059, almost the same as previous model
# Since all the variables were normalized before training, I can simply compare the coefficients of
# the model to see the most important variable
lreg2.coef_
# It turns out that this additional variable Mortgage doesn't help much with the modeling, even though
# it's a common belief that mortgage rates affect the house prices a lot. 
# One reason could be that there's not enough data across different time duration capturing the sales
# of houses. There are just a few unique combination of Year-Month of house sale and for each time
# duration, the SalePrices vary a lot for different houses. Mortgage alone can't have a strong impact
# on the housing prices.



## 8. Permutation tests
# function to calculate the score rmse log
def rmselog(pred, y):
    pred = np.log(pred)
    y = np.log(y)
    score = np.sqrt(mean_squared_error(y, pred))
    return score

# function to perform permutation test with 1000 permutations
def permutation_test(x, y):
    scores = []
    for i in range(1000):
        y = np.random.permutation(y)
        lreg = LinearRegression().fit(x, y)
        pred = lreg.predict(x)
        score = rmselog(pred, y)
        scores.append(score)
    return scores

# permutation test without fitting the model
def permutation_test2(pred, y):
    scores = []
    for i in range(1000):
        y = np.random.permutation(y)
        score = rmselog(pred, y)
        scores.append(score)
    return scores


# a. GrLivArea
xh = np.array(train['GrLivArea'])
xh = np.reshape(xh, (1460,1))
yh = train['SalePrice']
lregh = LinearRegression().fit(xh, yh)
lregh.score(xh, yh)
predh = lregh.predict(xh)
scoreh = rmselog(predh, yh) # empirical value
# pvalue determined by the fraction of scores in the permutation tests that are as extreme (low) as 
# the empirical value
# p = no of scores <= emp value / total no of scores
# get the list of scores
scores = permutation_test(xh, yh)
# now calculating the pvalue
pvalue = np.sum(scores <= scoreh)/len(scores)

# now performing the permutation tests for the other 9 variables 
variables = ['BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
             'TotRmsAbvGrd', 'GarageArea', 'YrSold', 'LotArea', 'OverallCond']
pvalues = dict()
yh = train['SalePrice']
for var in variables:
    xh = np.array(train[var])
    xh = np.reshape(xh, (1460,1))
    lregh = LinearRegression().fit(xh, yh)
    predh = lregh.predict(xh)
    scoreh = rmselog(predh, yh) # empirical value
    scores = permutation_test(xh, yh)
    # now calculating the pvalue
    pvalue = np.sum(scores <= scoreh)/len(scores)
    pvalues[var] = pvalue

pvalues = pd.DataFrame(list(pvalues.items()), columns=['Variables', 'pvalues'])
print(pvalues)
# Out of all these variables, only two variables had pvalues > 0.05
# OverallCond and YrSold, so the models built using these two variables are not very significant



## 9. Build a strong model for Kaggle submission
# First combine both train and test for common preprocessing
data = train.iloc[:, :-2].append(test, ignore_index=True)
# convert all categorical variables into dummy variables
tempo = pd.get_dummies(data)
# check for na values
nacols = tempo.columns[tempo.isna().any()].tolist()
# 11 columns with na values
np.sum(tempo[nacols[0]].isna())
# Many places have just 1 NAs, replace with mean where possible
tempo[nacols[0]]
tempo[nacols[0]].value_counts()
tempo[nacols[1]][tempo[nacols[1]].isna()] = tempo[nacols[1]].median(skipna=True)
tempo[nacols[2]][tempo[nacols[2]].isna()] = tempo[nacols[2]].mean(skipna=True)
tempo[nacols[3]][tempo[nacols[3]].isna()] = tempo[nacols[3]].median(skipna=True)
tempo[nacols[4]][tempo[nacols[4]].isna()] = tempo[nacols[4]].median(skipna=True)
tempo[nacols[5]][tempo[nacols[5]].isna()] = tempo[nacols[5]].mean(skipna=True)
tempo[nacols[6]][tempo[nacols[6]].isna()] = tempo[nacols[6]].mean(skipna=True)
tempo[nacols[7]][tempo[nacols[7]].isna()] = tempo[nacols[7]].median(skipna=True)
# GarageYrBlt is nan in some places, it could be because in some houses where there is
# no garage, garage year built is unnecessary. For the purpose of model building and not
# introducing much noise, I will impute the year with its corresponding YearBuilt date
# as the date has to be either that or after that date
tempo[nacols[8]][tempo[nacols[8]].isna()] = tempo['YearBuilt'][tempo[nacols[8]].isna()]
# GarageCars is also not applicable for the same houses. Can impute 0 values in their place
tempo[nacols[9]][tempo[nacols[9]].isna()] = 0
# similarly, the garage area could also be imputed as 0
tempo[nacols[10]][tempo[nacols[10]].isna()] = 0
# For LotFrontage, it's possible that there is no linear street connected to the property
# in that case, LotFrontage value can be imputed as 0
tempo[nacols[0]][tempo[nacols[0]].isna()] = 0

tempo = tempo.drop(columns='Id')
# can also include the d_index, first add AgeOfHouse as done earlier
tempo['AgeOfHouse'] = 2019 - tempo['YearBuilt']

# now can normalize all the variables
tempo = (tempo - tempo.min()) / (tempo.max() - tempo.min())

tempo['d_index'] = (np.exp(0.7086)*tempo['GrLivArea'] + 
                        np.exp(0.6234)*tempo['GarageArea'] +
                        np.exp(0.6136)*tempo['TotalBsmtSF'] +
                        np.exp(0.6058)*tempo['1stFlrSF'] - 
                        np.exp(0.624)*tempo['AgeOfHouse'])

tempo['d_index'] = ((tempo['d_index'] - tempo['d_index'].min())/
                       (tempo['d_index'].max() - tempo['d_index'].min()))

# split the data into train test
trainData = tempo.iloc[:1460, :]
testData = tempo.iloc[1460:, :]

# randomForest Regressor
regr = RandomForestRegressor(n_estimators=500, random_state=519)
regr.fit(trainData, train['SalePrice'])
regr.score(trainData, train['SalePrice'])
# Rsquared = 0.98
trpred = regr.predict(trainData)
trscore = rmselog(trpred, train['SalePrice'])
# 0.0594 

pred = regr.predict(testData)

submission = pd.DataFrame({'Id':list(range(1461, 2920)), 'SalePrice':pred})
submission.to_csv('submission.csv', index=False)

# Kaggle submission statistics (October 20, 2019):
# Rank - 2707/4795 (Top 57%)
# Score - 0.14473 (RMSE of log(price))
# Number of entries - 3



