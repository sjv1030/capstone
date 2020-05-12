# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:47:30 2020

@author: sjv1030_hp
"""

import os
import json
import pandas as pd
import numpy as np
from functions import *

# get model outputs 
os.chdir('../data')
with open('model_results.json') as f:
    scores = json.load(f)
with open('forecast_dict.json') as f:
    position = json.load(f)
spread_dt_df2 = pd.read_csv('spread_dt_df.csv')
spread_exp_df2 = pd.read_csv('spread_exp_df.csv')
pred_cycle_df2 = pd.read_csv('pred_cycle_df.csv')
os.chdir('../scripts')

# only keep positions where models have an accuracy score above 50%
fcst = pd.DataFrame.from_dict(scores,orient='index')
fcst.columns = ['model','score']
fcst = fcst.loc[fcst['score'] > 0.5]
position_df = pd.DataFrame.from_dict(position)

# subset by spreads where models have scores above 50%
position_df = position_df[fcst.index] 

# add a datetime index to the expansion position dataframe
position_df.index = pd.date_range(start='2017-01-01', 
                           periods=position_df.shape[0], freq='MS')

spread_dt_df2['Date'] = pd.to_datetime(spread_dt_df2['Date'])
spread_dt_df2.set_index('Date',inplace=True)

if spread_exp_df2.columns[0].lower() not in ['date']:
    c = list(spread_exp_df2.columns)
    c.pop(0)
    c.insert(0,'Date')
    spread_exp_df2.columns = c
spread_exp_df2['Date'] = pd.to_datetime(spread_exp_df2['Date'])
spread_exp_df2.set_index('Date',inplace=True)

# subset by spreads where models have scores above 50%
spread_exp_df2 = spread_exp_df2[fcst.index]

# merge all spreads into one dataframe to calculate daily log returns
ret_df = pd.concat([spread_dt_df2,spread_exp_df2],axis=1)

# resample daily price spreads into monthly using last
ret_df_m = ret_df.resample('MS').last()

# calculate monthly log returns
ret_df_m = np.log(ret_df_m/ret_df_m.shift(1))

# return dataframe should start in 2017
ret_df_m = ret_df_m.loc['2017'::] 


# prepare dataframes for merging
pred_cycle_df2.columns = ['date','cycle']
pred_cycle_df2['date'] = pd.to_datetime(pred_cycle_df2['date'])
pred_cycle_df2.set_index('date',inplace=True)

# subset the cyclical expansion forecast dataframe for select dates
pred_cycle_df2 = pred_cycle_df2.loc[position_df.index]

# multiply position dataframe with return dataframe
ret_exp_matrix = ret_df_m.loc[position_df.index,position_df.columns]
bt_returns = position_df * ret_exp_matrix
bt_returns = pd.concat([bt_returns,
                        ret_df_m.loc[position_df.index,
                                     spread_dt_df2.columns]],axis=1)

# iterate through cycle forecasts
# if model predicts a contraction, then output equal-weighted return of
# spreads identified to do well during a contraction.
# if model predicts an expansion, then output equal-weighted return of
# spreads forecasted to do well

backtest_df = pd.concat([pred_cycle_df2,bt_returns],axis=1)
backtest_df.dropna(inplace=True)

# backtest all
backtest_df['bt'] = np.where(backtest_df['cycle']==1,
                             backtest_df[spread_dt_df2.columns].mean(axis=1),
                             backtest_df[spread_exp_df2.columns].mean(axis=1))
    
# calculate annualized mean return and annualized standard deviation
# to calculate sharpe ratio with a risk-free return = 0
# save to dictionary and dataframe

mu_bt = round(backtest_df['bt'].mean()*12*100,2)
vol_bt = round(backtest_df['bt'].std()*np.sqrt(12)*100,2)
sharpe_bt = round(mu_bt/vol_bt,2)

bt_dict = dict()
bt_dict['all'] = [mu_bt,vol_bt,sharpe_bt]

bt_df = backtest_df[spread_dt_df2.columns].copy()
bt_df['cycle'] = backtest_df['cycle']
for _ in bt_df.columns:
    if _ in ['cycle']: 
        continue
    else:
        bt_df[_+'_bt'] = np.where(bt_df['cycle']==1,
                                  bt_df[_],bt_df[_]*-1)
        bt_dict[_] = [round(bt_df[_].mean()*12*100,2),
                      round(bt_df[_].std()*np.sqrt(12)*100,2),
                      round(
                        (bt_df[_].mean()*12*100)/(bt_df[_].std()*np.sqrt(12)*100)
                          ,2)]

final_df = pd.DataFrame.from_dict(bt_dict,
                                  columns=['ann_ret','ann_vol','sharpe'],
                                  orient='index').sort_values(by=['sharpe'],
                                                              ascending=False)

# export results as a csv file
os.chdir('../data')
final_df.to_csv('final.csv')
os.chdir('../scripts')