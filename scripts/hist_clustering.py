# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:58:43 2020

@author: sjv1030_hp
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.cluster import KMeans

os.chdir('../scripts')

# set font size for all charts
plt.rcParams.update({'font.size': 22})

# get market price data
tbl_list = print_tables('mkt_prices')

# store closing prices in a dictionary
prices_dict = dict()
for _ in tbl_list:
    tmp = get_table(_,'mkt_prices')
    tmp = tmp[['Date','Close']]
    tmp.columns = ['Date',_]
    prices_dict[_] = tmp
    
# skip duplicates and ETFs that didn't have price history back then
skip_tickers = ['RTY','SP500','DOW','NDX','VXX','XLC','XLRE',
                'FXI','JNK','EWW','EWZ','EFA','TIP','SHY']

# store the percent return during the recession
ret_df = pd.DataFrame()
col_names = []
for _ in prices_dict.keys():
    if _ in skip_tickers:
        continue
    elif _ == 'SPY':
        tmp = prices_dict[_].copy()
        ret_df = pd.concat([ret_df,tmp],axis=1)
    else:
        col_names.append(_)
        tmp = prices_dict[_][_].copy()
        ret_df = pd.concat([ret_df,tmp],axis=1)
ret_df.set_index('Date',inplace=True)


######################################################
######### Cluster Analysis for the GFC ###############
######################################################
# create a dataframe to store percent return and daily volatility
gfc = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2007-10-09','2009-04-01')
    gfc = gfc.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
gfc['sharpe'] =  gfc['return']/gfc['sigma']
gfc.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
label = list(np.repeat([1,2,3],10))
label.append(3)
gfc['class'] = label
gfc.dropna(inplace=True)

# scatter plot of return and volatility
ax = gfc.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(gfc['ticker']):
    ax.annotate(lbl, (gfc['sigma'].iloc[i]+0.05, gfc['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the Global Financial Crisis',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

# Using some of the code in the below website, we created a clustering chart
# source:
# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(gfc[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - Global Financial Crisis',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 4 clusters, as 4 clusters ignores outliers
# red circles mark the clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(gfc[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(gfc['sigma'], gfc['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(gfc['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(gfc['return']))-2, int(np.max(gfc['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the Global Financial Crisis',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()


##############################################################
#### Cluster Analysis for the Cyclical Slowdown in Mid 2010 ######
##############################################################
# create a dataframe to store percent return and daily volatility
slow10 = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2010-07-01','2010-09-01')
    slow10 = slow10.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
slow10['sharpe'] =  slow10['return']/slow10['sigma']
slow10.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
slow10['class'] = label
slow10.dropna(inplace=True)


# scatter plot of return and volatility
ax = slow10.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(slow10['ticker']):
    ax.annotate(lbl, (slow10['sigma'].iloc[i]+0.05, slow10['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the 2010 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(slow10[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - 2010 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 4 clusters
# red circles mark the clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(slow10[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(slow10['sigma'], slow10['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(slow10['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(slow10['return']))-2, int(np.max(slow10['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the 2010 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()

##############################################################
#### Cluster Analysis for the Cyclical Expansion in Late 2010 ######
##############################################################
# create a dataframe to store percent return and daily volatility
exp10 = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2010-09-01','2011-08-01')
    exp10 = exp10.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
exp10['sharpe'] =  exp10['return']/exp10['sigma']
exp10.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
exp10['class'] = label
exp10.dropna(inplace=True)
exp10 = exp10.loc[exp10['ticker']!='SLV']


# scatter plot of return and volatility
ax = exp10.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(exp10['ticker']):
    ax.annotate(lbl, (exp10['sigma'].iloc[i]+0.05, exp10['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the 2010-11 Cyclical Expansion',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(exp10[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - 2010-11 Cyclical Expansion',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 4 clusters
# red circles mark the clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(exp10[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(exp10['sigma'], exp10['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(exp10['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(exp10['return']))-2, int(np.max(exp10['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the 2010-11 Cyclical Expansion',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()

##############################################################
#### Cluster Analysis for the Cyclical Slowdown in Late 2011 ######
##############################################################
# create a dataframe to store percent return and daily volatility
slow12 = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2011-08-01','2011-12-01')
    slow12 = slow12.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
slow12['sharpe'] =  slow12['return']/slow12['sigma']
slow12.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
slow12['class'] = label
slow12.dropna(inplace=True)


# scatter plot of return and volatility
ax = slow12.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(slow12['ticker']):
    ax.annotate(lbl, (slow12['sigma'].iloc[i]+0.05, slow12['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the 2012 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(slow12[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - 2012 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 4 clusters, as 4 clusters ignores outliers
# red circles mark the clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(slow12[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(slow12['sigma'], slow12['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(slow12['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(slow12['return']))-2, int(np.max(slow12['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the 2012 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()

##############################################################
#### Cluster Analysis for the Cyclical Expansion 2012 - 2015 ######
##############################################################
# create a dataframe to store percent return and daily volatility
exp12 = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2011-12-01','2015')
    exp12 = exp12.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
exp12['sharpe'] =  exp12['return']/exp12['sigma']
exp12.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
exp12['class'] = label
exp12.dropna(inplace=True)


# scatter plot of return and volatility
ax = exp12.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(exp12['ticker']):
    ax.annotate(lbl, (exp12['sigma'].iloc[i]+0.05, exp12['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the 2012-15 Cyclical Expansion',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(exp12[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - 2012-15 Cyclical Expansion',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 4 clusters
# red circles mark the clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(exp12[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(exp12['sigma'], exp12['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(exp12['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(exp12['return']))-2, int(np.max(exp12['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the 2012-15 Cyclical Expansion',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()

##############################################################
#### Cluster Analysis for the Cyclical Slowdown in Early 16 ######
##############################################################
# create a dataframe to store percent return and daily volatility
slow15 = pd.DataFrame(columns=['ticker','return','sigma'])
for _ in ret_df.columns:
    # for each ticker, send closing prices, beg and end dates, and 
    # return two floats: one for the return and another for volatility
    r, s = get_return(ret_df[[_]],'2016-01-20','2016-03-01')
    slow15 = slow15.append(pd.DataFrame([[_,r,s]],columns=['ticker','return','sigma']))

# calculate a sharpe ratio with a 0% risk-free rate
slow15['sharpe'] =  slow15['return']/slow15['sigma']
slow15.sort_values('sharpe',inplace=True)
# add arbitary classes splitting it roughly equally
slow15['class'] = label
slow15.dropna(inplace=True)


# scatter plot of return and volatility
ax = slow15.plot.scatter(x='sigma', y='return', s=150,color='red',figsize=(20,15))
for i, lbl in enumerate(slow15['ticker']):
    ax.annotate(lbl, (slow15['sigma'].iloc[i]+0.05, slow15['return'].iloc[i]),size=20)
plt.title('Return vs. Risk\nDuring the Early 2016 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.axhline(y=0,color='k',linestyle=':')
plt.show()    

wcss = [] # a list to store sum of squared distances between clusters
# try different number of clusters and loop through 
# while collecting sum of squared distances
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)
    kmeans.fit(slow15[['sigma','return']] )
    wcss.append(kmeans.inertia_)
# create an "elbow" plot to identify how many clusters are needed
plt.figure(figsize=(20,15))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method - 2016 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Number of clusters',fontweight='bold')
plt.ylabel('Within Cluster Sum of Squared Differences',fontweight='bold')
plt.show()

# create a scatter plot with 3 clusters, as 3 clusters ignores outliers
# red circles mark the clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
pred_y = kmeans.fit_predict(slow15[['sigma','return']] )
plt.figure(figsize=(20,15))
plt.scatter(slow15['sigma'], slow15['return'],s=300)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=2000, c='red')
plt.xticks(np.arange(0, np.max(slow15['sigma'])+1, 1.0))
plt.yticks(np.arange(int(np.min(slow15['return']))-2, int(np.max(slow15['return']))+2, 12))
plt.axhline(y=0,linestyle=':',color='k')
plt.title('Return vs. Risk\nDuring the Early 2016 Cyclical Slowdown',fontweight='bold')
plt.xlabel('Volatility %',fontweight='bold')
plt.ylabel('Return %',fontweight='bold')
plt.show()
