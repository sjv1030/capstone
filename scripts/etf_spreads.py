# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:36:55 2020

@author: sjv1030_hp
"""
import os
import json
import pandas as pd
import numpy as np
from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# import ETF prices and calculate spreads and show stationarity

# get historical market data
etfs = ["SPY","QQQ","DIA","EEM","IWM","IWO","IWF",
        "GLD","GDX","SLV",
        "TLT","LQD","HYG","MBB","MUB",
        "XLF","XBI","XLV","XRT","XLP","XLY","XLU","XLK","XLC","XLE","XOP",
        "OIH","XME","ITB","KRE","SMH","XLI"]

fin_index = dict()
for _ in etfs:
    fin_index[_] = get_table(_,'mkt_prices')  


# spreads during downturns
spread_dt_lst = ['GLDSLV','TLTSPY','TLTHYG','XLPXLY','XLUXLF','TLTLQD','TLTMBB',
              'GLDOIH','SLVXME','TLTKRE','XLEXOP','QQQEEM','XLPXRT','HYGXOP',
              'GDXXME','SPYIWM']

# create a dictionary of spreads
spread_dt = dict()
for _ in spread_dt_lst:
    tmp = fin_index[_[:3]][['Date','Close']].copy()
    tmp['Close'].fillna(method='ffill',inplace=True)
    tmp2 = fin_index[_[3:]][['Date','Close']].copy()
    tmp2['Close'].fillna(method='ffill',inplace=True)
    tmp.set_index('Date',inplace=True)
    tmp2.set_index('Date',inplace=True)
    tmp3 = pd.concat([tmp,tmp2],axis=1).dropna()
    tmp3.columns = [_[:3],_[3:]]
    tmp3[_] = tmp3[_[:3]] / tmp3[_[3:]]
    spread_dt[_] = tmp3[[_]]

# merge all spreads to trade during a downturn
spread_dt_df = spread_dt['TLTSPY'].copy()
for k,v in spread_dt.items():
    if k in ['TLTSPY']: continue
    print(k)
    spread_dt_df = pd.concat([spread_dt_df,v],axis=1)
spread_dt_df.index = pd.to_datetime(spread_dt_df.index)
# export as a csv file
os.chdir('../data')
spread_dt_df.to_csv('spread_dt_df.csv')
os.chdir('../scripts')
    
# spreads during expansions
spread_exp_lst = ['GDXGLD','SPYTLT','KREXLF','XOPXLE','XLYXLP','XLIXLU','XBIXLV',
                  'XLVSPY','XLKSPY','XLESPY','HYGTLT','XRTXLP','ITBKRE',
                  'IWMSPY','MBBTLT','IWFSPY','IWOIWM','LQDTLT','XMEGDX']

# create a dictionary of spreads
spread_exp = dict()
for _ in spread_exp_lst:
    tmp = fin_index[_[:3]][['Date','Close']].copy()
    tmp2 = fin_index[_[3:]][['Date','Close']].copy()
    tmp.set_index('Date',inplace=True)
    tmp2.set_index('Date',inplace=True)
    tmp3 = pd.concat([tmp,tmp2],axis=1).dropna()
    tmp3.columns = [_[:3],_[3:]]
    tmp3[_] = tmp3[_[:3]] / tmp3[_[3:]]
    spread_exp[_] = tmp3[[_]]

# test if spread 1-month returns are stationary in order to try to forecast
spread_exp_trade = dict()
for k in spread_exp.keys():
    if stationary(spread_exp[k],k):
        spread_exp_trade[k] = spread_exp[k]
    else:
        print(k)

# merge all spreads to trade during an expansion
spread_exp_df = spread_exp['SPYTLT'].copy()
for k,v in spread_exp_trade.items():
    if k in ['SPYTLT']: continue
    print(k)
    spread_exp_df = pd.concat([spread_exp_df,v],axis=1)
spread_exp_df.index = pd.to_datetime(spread_exp_df.index)
# export as a csv file
os.chdir('../data')
spread_exp_df.to_csv('spread_exp_df.csv')
os.chdir('../scripts')

# bring in macro factors to try to forecast spread 1-month returns
# this will only be done for spreads to trade during expansions
# spreads during recessions don't need to be forecasted given the strategy

# get macro data
rf_list = print_tables('factors')

# get macro dictionary
os.chdir('../data')
with open('macro_dict.json') as f:
    rf_key_dict = json.load(f)
os.chdir('../scripts')

# get macro factors
rf = pd.DataFrame()
for _ in rf_list:
    tmp = get_table(_,'factors') 
    tmp = tmp[['date','yoy','ratio6ma','yoy6ma']]
    tmp['date'] = pd.to_datetime(tmp['date'])
    tmp.set_index('date',inplace=True)
    tmp.columns = [_+'_'+x for x in tmp.columns]
    tmp = tmp.loc['2005'::]
    rf = pd.concat([rf,tmp],axis=1)


# cleanup macro factors dataframe
# drop macro factors that are missing 20% or more
rf_cleanup = pd.DataFrame(rf.isna().sum().sort_values(ascending=False)/rf.shape[0]*100)
rf_cleanup.columns = ['pct_NA']
rf2 = rf[rf_cleanup.loc[rf_cleanup['pct_NA']<20].index]

# get downturns
os.chdir('../data')
dt_df = pd.read_csv('downturns.csv')
os.chdir('../scripts')
dt_df['Date'] = pd.to_datetime(dt_df['Date'])
dt_df.set_index('Date',inplace=True)
print(dt_df.head())

# iterate through each spread for expansionary time period
# subset into three groups: train, validate, test
# change target to 0 and 1
# where 0 = negative return and 1 = positive return
model_results = dict()
forecast_dict = dict()
for _ in spread_exp.keys():
    print(_)
    tmp_df = spread_exp[_].copy()
    tmp_df.index = pd.to_datetime(tmp_df.index)
    tmp_df = tmp_df.resample('MS').last()[1:]
    tmp_df['mom'] = tmp_df[_].pct_change()*100
    tmp_df[_+'_tgt'] = np.where(tmp_df['mom'] > 0,1,0) 
    tmp_df.dropna(inplace=True)
    full = pd.concat([tmp_df[_+'_tgt'],rf2.shift(2)],axis=1)
    # only use datapoints during an expansionary cycle
    full = full.loc[dt_df.loc[dt_df['target'] == 0].index]
    full.dropna(inplace=True)   

    # breakup dataset to train the model on 70% of data
    # validate the model on 20% of data
    # and test the model on 10% of data
    train_cutoff = int(full.shape[0]*0.7)
    train_valid = int(full.shape[0]*0.2)
    test_cutoff = full.shape[0] - train_cutoff - train_valid

    train_df = full.iloc[:train_cutoff,:]
    valid_df = full.iloc[train_cutoff:train_cutoff+train_valid,:]
    test_df = full.iloc[-test_cutoff:,:]

    train_X = train_df.iloc[:,1:].copy() # features
    train_y = train_df.iloc[:,0].copy() # target

    valid_X = valid_df.iloc[:,1:].copy() # features
    valid_y = valid_df.iloc[:,0].copy() # target

    test_X = test_df.iloc[:,1:].copy() # features
    test_y = test_df.iloc[:,0].copy() # target

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=628)
    rfc.fit(train_X, train_y)  
    
    feature_importances = pd.DataFrame(rfc.feature_importances_,
                    index = train_X.columns,
            columns=['importance']).sort_values('importance',ascending=False)
    
    #print(feature_importances.head(10))
    
    #rfc_disp = plot_roc_curve(rfc, valid_X, valid_y)
    #plt.show() # plot AUC curves
    
    #disp = plot_confusion_matrix(rfc, valid_X, valid_y,
                #display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    #disp.ax_.set_title('Confusion Matrix for '+_)
    #print(_)
    #print(disp.confusion_matrix)
    
    pred = rfc.predict(valid_X)
    #print(classification_report(valid_y,pred))
    rfc_score = rfc.score(valid_X,valid_y)
    
    #print('---------ATTEMPT NUMBER 2------------')
    
    rfc2 = RandomForestClassifier(n_estimators=100, random_state=628)
    rfc2.fit(train_X[feature_importances.index[:10]], train_y)  
    
    feature_importances = pd.DataFrame(rfc2.feature_importances_,
                    index = feature_importances.index[:10],
            columns=['importance']).sort_values('importance',ascending=False)
    
    #print(feature_importances.head(10))
    
    #rfc_disp = plot_roc_curve(rfc2, valid_X[feature_importances.index[:10]], valid_y)
    #plt.show() # plot AUC curves
    
    #disp = plot_confusion_matrix(rfc2, valid_X[feature_importances.index[:10]], valid_y,
    #            display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    #disp.ax_.set_title('Confusion Matrix for '+_)
    #print(_)
    #print(disp.confusion_matrix)
    
    pred = rfc2.predict(valid_X[feature_importances.index[:10]])
    rfc2_score = rfc2.score(valid_X[feature_importances.index[:10]],valid_y)  
   
    if rfc2_score > rfc_score: 
        rfs = rf2.shift(2).loc['2017'::].dropna()[:-1]
        forecast = rfc2.predict(rfs[feature_importances.index[:10]])
        forecast_dict[_] = list(forecast)
        model_results[_] = ('rfc2',rfc2_score)
    else:
        rfs = rf2.shift(2).loc['2017'::].dropna()[:-1]
        forecast = rfc.predict(rfs)
        forecast_dict[_] = list(forecast)
        model_results[_] = ('rfc',rfc_score)
               
# save model scores and forecasts
os.chdir('../data')
with open('model_results.json', 'w') as f:
    json.dump(model_results, f)
with open('forecast_dict.json', 'w') as f:
    json.dump(forecast_dict, f)

