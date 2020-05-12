# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:34:57 2020

@author: sjv1030_hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
plt.rcParams.update({'font.size': 22})

from functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# CREATE FINANCIAL VULNERABILITY INDEX
# from database, bring in 
# 1) market indices
# 2) VIX
# 3) US 10s2s, US10s3M, TED spread, BBB Spread, AAA Spread, CPFF
# 4) US trade-weighted dollar index
# calculate changes for each and calculate PCA

fin_index = dict()
fin_index['vix'] = get_table('VIXCLS','factors')
fin_index['sp500'] = get_table('SP500','mkt_prices')
fin_index['rty'] = get_table('RTY','mkt_prices') 
fin_index['us102'] = get_table('T10Y2Y','factors')
fin_index['us103'] = get_table('T10Y3M','factors')  
fin_index['ted'] = get_table('TEDRATE','factors') 
fin_index['baa'] = get_table('BAA10Y','factors')
fin_index['aaa'] = get_table('AAA10Y','factors')
fin_index['cp'] = get_table('CPFF','factors')
fin_index['twi'] = get_table('DTWEXBGS','factors')
fin_index['twiem'] = get_table('DTWEXEMEGS','factors')

fin_ts = dict()
for k in fin_index.keys():
    tmp = fin_index[k]
    if k in ['sp500','rty']:
        tmp.columns = ['date','close']
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp.set_index(tmp['date'],inplace=True)
        tmp = tmp.resample('MS').last()
        tmp['mom'] = tmp['close'].pct_change()*100
    else:
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp.set_index(tmp['date'],inplace=True)
    fin_ts[k] = tmp.loc['2000'::]

fin = fin_ts['us102'][['mom']]
fin_col_list = ['us102']
for k in fin_ts.keys():
    if k in ['us102']: continue
    fin = pd.concat([fin,fin_ts[k]['mom']],axis=1)
    fin_col_list.append(k)
fin.columns = fin_col_list

# standardize the data
fin_factors = StandardScaler().fit_transform(fin.loc[:'2016'].dropna().values)
# get PCA class and apply to features
fin_pca = PCA()
prinComp = fin_pca.fit_transform(fin_factors)

# create a plot showing how many principal components 
# explain about 70% of the variance
plt.figure(figsize=(12,10))
plt.plot([str(i) for i in range(1,12)],np.cumsum(fin_pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.title('Cumulative Explained Variance of Leading Financial Variables')
plt.axhline(y=0.7,color='r',linestyle=':')
plt.axvline(x=2.0,color='r',linestyle=':')
plt.show()

# judging by the chart, 3 principal components will suffice
fin_pc_df = pd.DataFrame(prinComp[:,:3], columns = ['PC1', 'PC2','PC3'])
fin_pc_df.index = fin.loc[:'2016'].dropna().index


# CREATE MACRO VULNERABILITY INDEX
# from database, bring in 
# 1) Financial Conditions Index
# 2) Building Permits
# 3) Empire Manufacturing Index
# 4) Philadelphia Manufacturing Index
# 5) US 5-year Inflation Expectations
# 6) University of Michigan Consumer Sentiment
# 7) US Retail Sales
# 8) US Employment to Population Ratio
# 9) Index of Aggregate Weekly Hours
# calculate changes relevant for each


macro_index = dict()
macro_index['fci']  = get_table('NFCI','factors')
macro_index['bp'] = get_table('PERMIT','factors')
macro_index['empire'] = get_table('GACDISA066MSFRBNY','factors')
macro_index['philly'] = get_table('GACDFSA066MSFRBPHI','factors')
macro_index['infexp'] = get_table('T5YIE','factors')
macro_index['umich'] = get_table('UMCSENT','factors')
macro_index['retail'] = get_table('RSAFS','factors')
macro_index['emp'] = get_table('EMRATIO','factors')
macro_index['hours'] = get_table('AWHI','factors')

for k in macro_index.keys():
    tmp = macro_index[k]
    tmp['date'] = pd.to_datetime(tmp['date'])
    tmp.set_index('date',inplace=True)

macro = pd.DataFrame()
macro_col_list = list()
for k in macro_index.keys():
    tmp = macro_index[k].copy()
    if k in ['infexp']:
        macro = pd.concat([macro,tmp[['T5YIE']].diff()],axis=1)
    elif k in ['empire','philly']:
        macro = pd.concat([macro,tmp[['value']].diff()],axis=1)
    elif k in ['emp']:
        macro = pd.concat([macro,tmp[['value']]],axis=1)
    else:
        macro = pd.concat([macro,tmp[['value']].pct_change(1)],axis=1)
    macro_col_list.append(k)
macro.columns = macro_col_list
macro.index = pd.to_datetime(macro.index)

# standardize the data
macro_factors = StandardScaler().fit_transform(macro.loc['2005':'2016'].dropna().values)
# get PCA class and apply to features
macro_pca = PCA()
prinComp2 = macro_pca.fit_transform(macro_factors)

# create a plot showing how many principal components 
# explain about 80% of the variance
plt.figure(figsize=(12,10))
plt.plot([str(i) for i in range(1,10)],np.cumsum(macro_pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.title('Cumulative Explained Variance of Leading Macro Variables')
plt.axhline(y=0.70,color='r',linestyle=':')
plt.axvline(x=4.0,color='r',linestyle=':')
plt.show()

# judging by the chart, 5 principal components will suffice
macro_pc_df = pd.DataFrame(prinComp2[:,:5], columns = ['PC1', 'PC2','PC3','PC4','PC5'])
macro_pc_df.index = macro.loc['2005':'2016'].dropna().index

# Use all 5 principal components to identify cyclical contraction/expansion
pc_df = pd.concat([fin_pc_df,macro_pc_df],axis=1)
pc_df.dropna(inplace=True)
pc_df.columns = ['PC'+str(i) for i in range(1,pc_df.shape[1]+1)]
pc_df = pc_df.shift(2)

# get market prices for Russell 2000 and S&P 500
indices = ["^RUT","^GSPC"]
mkt_indices = yf.download(
        tickers = indices,
        period = "15y",
        interval = "1d",
        group_by = 'Ticker',
        # adjust all OHLC automatically
        auto_adjust = True)

sp = mkt_indices['^GSPC']
r2k = mkt_indices['^RUT']

# create target variable
downturns = dict()
downturns['spx'] = label_downturn(sp.loc[:'2016'])
downturns['r2k'] = label_downturn(r2k.loc[:'2016'])
        
downturn = pd.concat([downturns['spx'],downturns['r2k']],axis=1)
downturn.columns = ['spx','r2k']
downturn['target'] = downturn[['spx','r2k']].max(axis=1)
print(downturn.head())

print(downturn['spx'].value_counts())
print(downturn['r2k'].value_counts())
print(downturn['target'].value_counts())

downturn['target'].value_counts().plot(kind='bar',rot=1)
plt.title('Number of Contractions (1) and Expansions (0)\nBetween 2005 and 2016')
plt.show()

plt.figure(figsize=(12,10))
plt.title('Cycle Variable:\n1 = Contraction | 0 = Expansion')
downturn['target'].plot()
plt.show()

# export downturns to save as a CSV file
os.chdir('../data')
downturn[['target']].to_csv('downturns.csv')
os.chdir('../scripts')

# combine target and features
dataset = pd.concat([downturn[['target']],pc_df],axis=1)
dataset.dropna(inplace=True) # drop NAs

# target variable is not balanced
print(dataset['target'].value_counts())
print('The dataset is not balanced:')
print('% not in contraction:',
      round(dataset['target'].value_counts()[0]/dataset.shape[0]*100,2))
print('% in contraction:',
      round(dataset['target'].value_counts()[1]/dataset.shape[0]*100,2))

sns.countplot(x='target',data=dataset)
plt.title('Balance of Target Variable')
plt.show()


# drop some rows where target is 0 to make the dataframe more balanced
dataset = dataset.loc[(dataset.index < pd.to_datetime('2016-03-01'))]
dataset = dataset.iloc[1:,:]
dataset = dataset.loc[(dataset.index == pd.to_datetime('2006-05-01')) |
                      (dataset.index > pd.to_datetime('2007-06-01'))]
dataset = dataset.loc[(dataset.index < pd.to_datetime('2012-07-01')) |
                      (dataset.index > pd.to_datetime('2014-09-01'))]
dataset = dataset.loc[(dataset.index < pd.to_datetime('2011-01-01')) |
                      (dataset.index > pd.to_datetime('2011-07-01'))]
dataset = dataset.loc[(dataset.index < pd.to_datetime('2014-11-01')) |
                      (dataset.index > pd.to_datetime('2015-07-01'))]

# the dataset is now more balanced
print(dataset['target'].value_counts())
print('The dataset is now more balanced:')
print('% not in contraction:',
      round(dataset['target'].value_counts()[0]/dataset.shape[0]*100,2))
print('% in contraction:',
      round(dataset['target'].value_counts()[1]/dataset.shape[0]*100,2))

sns.countplot(x='target',data=dataset)
plt.title('Balance of Target Variable')
plt.show()

# breakup dataset to train the model on 70% of data
# validate the model on 20% of data
# and test the model on 10% of data
train_cutoff = int(dataset.shape[0]*0.7)
train_valid = int(dataset.shape[0]*0.2)
test_cutoff = dataset.shape[0] - train_cutoff - train_valid

train_df = dataset.iloc[:train_cutoff,:]
valid_df = dataset.iloc[train_cutoff:train_cutoff+train_valid,:]
test_df = dataset.iloc[-test_cutoff:,:]

train_X = train_df.iloc[:,1:].copy() # features
train_y = train_df.iloc[:,0].copy() # target

valid_X = valid_df.iloc[:,1:].copy() # features
valid_y = valid_df.iloc[:,0].copy() # target

test_X = test_df.iloc[:,1:].copy() # features
test_y = test_df.iloc[:,0].copy() # target

# test four different classification models
# 1) Support Vector Classification
svc = SVC(kernel='linear',random_state=628)
svc.fit(train_X, train_y)
plt.figure(figsize=(12,10))
svc_disp = plot_roc_curve(svc, valid_X, valid_y)
plt.show()

# 2) Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=628)
rfc.fit(train_X, train_y)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, valid_X, valid_y, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show() # plot SVC and RFC AUC curves

# 3) Logistic Regression
lr = LogisticRegression(random_state=628)
lr.fit(train_X, train_y)
lr_disp = plot_roc_curve(lr, valid_X, valid_y)
plt.show()

# 4) Guassian Navie Bayes
nb = GaussianNB()
nb.fit(train_X, train_y)
ax = plt.gca()
nb_disp = plot_roc_curve(nb, valid_X, valid_y, ax=ax, alpha=0.8)
lr_disp.plot(ax=ax,alpha=0.8)
plt.show() # plot Logistic and NB AUC curves

# plot the AUC curves of the best two models
ax = plt.gca()
svc_disp.plot(ax=ax, alpha=0.8)
nb_disp.plot(ax=ax, alpha=0.8)
plt.show()

# print the confusion matrix for the top two models
# Plot non-normalized confusion matrix

for m in [svc,rfc,lr,nb]:
    disp = plot_confusion_matrix(m, valid_X, valid_y,
                display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    disp.ax_.set_title('Confusion Matrix:'+str(m))

    print(str(m))
    print(disp.confusion_matrix)
    
    p = m.predict(valid_X)
    print(classification_report(valid_y,p))
    
    total = np.sum(disp.confusion_matrix)
    accur = (disp.confusion_matrix[0][0] + disp.confusion_matrix[1][1])/total
    print('Accuracy:',str(round(accur,2)))
    print('Error rate:',str(round(1-accur,2)))
plt.show()

# logistic regression has better metrics on the validation dataset
# use logistic regression to forecast and compare versus test dataset
pred = lr.predict(test_X)
print(classification_report(test_y,pred))
final = plot_confusion_matrix(lr, test_X, test_y,
                display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
print(final.confusion_matrix)
pred_df = pd.DataFrame()
pred_df['pred'] = pred
pred_df['actual'] = test_y.values
pred_df.index = test_df.index
print(pred_df)

ax = pred_df.plot(kind='bar',rot=0,edgecolor='k')
ax.set_xticklabels(map(lambda x: xlbl_format(x), pred_df.index))
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_title('Cyclical Contraction\nModel Prediction vs. Actual')
plt.show()

# Use final model to forecast cyclical contraction/expansion beyond 2016
pred_cycle = dict()
for idx in fin.loc['2016':].index:
    pc = create_pca(fin.loc['2015':idx],macro.loc['2015':idx]) 
    pred_cycle[idx] = lr.predict(pc.dropna())[-1]
pred_cycle_df = pd.DataFrame.from_dict(pred_cycle,orient='index')
pred_cycle_df.index = pd.to_datetime(pred_cycle_df.index)

# export downturn predictions as a CSV file
os.chdir('../data')
pred_cycle_df.to_csv('pred_cycle_df.csv')
os.chdir('../scripts')
