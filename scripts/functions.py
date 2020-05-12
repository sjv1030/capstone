# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:47:02 2020

@author: sjv1030_hp
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

def get_table(tbl,db):
    """ 
    takes in a table name and database name as parameters
    returns a dataframe
    """
    os.chdir('../data')
    con = sqlite3.connect(db+'.db')
    out = pd.read_sql_query("SELECT * FROM " + tbl, con)
    os.chdir('../scripts')
    return out

def print_tables(db):
    """
    get tables from database and return as a list
    """
    os.chdir('../data')
    con = sqlite3.connect(db+'.db')
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    out = [i[0] for i in cursor.fetchall()]
    print('Table Names:')
    print(out)
    con.close()
    os.chdir('../scripts')
    return out

def get_return(values,beg_date,end_date):
    """
    get return and vol for a timeseries for the beg and end date
    """
    beg = values.loc[beg_date:end_date].head(1).values[0][0]
    end = values.loc[beg_date:end_date].tail(1).values[0][0]
    ret = (end/beg-1)*100
    sigma = values.loc[beg_date:end_date].pct_change().std().values[0]*100    
    return ret, sigma

def create_ts(ts,td='2020-02-01'):
    ts['realtime_start'] = pd.to_datetime(ts['realtime_start'])
    ts['date'] = pd.to_datetime(ts['date'])
    ts_df = ts.loc[ts['date']>'1999-12-31'].copy()
    ls_dates = ts_df['date'].drop_duplicates()
    td = pd.to_datetime(td)
    ls_ = [x for x in ls_dates[12:] if x < td ]
    output = pd.DataFrame()
    for i, ls_dt in enumerate(ls_):
        ts_ = ts_df.loc[(ts_df['date']<=ls_dt)].copy()
        if ts_df.empty: break
        row1 = ts_[['realtime_start','date','value']].loc[ts_['date']==ls_dt+pd.DateOffset(months=-12)]
        row1 = row1.loc[row1['realtime_start']<ls_dt+pd.DateOffset(months=2)]
        row2 = ts_[['realtime_start','date','value']].loc[ts_['date']==ls_dt+pd.DateOffset(months=-1)]
        row2 = row2.loc[(row2['realtime_start']<ls_dt+pd.DateOffset(months=2)) & 
                        (row2['realtime_start']>ls_dt+pd.DateOffset(months=1))]
        row3 = ts_[['realtime_start','date','value']].loc[ts_['date']==ls_dt].head(1)
        tmp_df = pd.DataFrame()
        tmp_df = tmp_df.append(pd.DataFrame(row1,columns=['date','value']))
        tmp_df = tmp_df.append(pd.DataFrame(row2,columns=['date','value']))
        tmp_df = tmp_df.append(pd.DataFrame(row3,columns=['date','value']))
        tmp_df['yoy'] = tmp_df['value'].pct_change(2)*100
        tmp_df['delta'] = tmp_df['value'].diff()
        tmp_df['mom'] = tmp_df['value'].pct_change()*100
        row_list = tmp_df[['date','value','delta','mom','yoy']].tail(1)
        output = output.append(row_list,
                               ignore_index=True)
    return output

def label_downturn(ts):
    """
    Takes a dataframe holding historical data of a financial market.
    Returns a monthly dataframe of a dummy variable where:
        0 = no cyclical contraction over the course of 30 calendar days
        1 = cylical contraction over the course of 30 calendar days

    Parameters
    ----------
    ts : dataframe

    Returns
    -------
    dataframe with one dummy variable

    """
    tmp = ts.copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp['yyyymm'] = tmp.index.year.astype(str)+tmp.index.month.astype(str)
    tmp['lowest'] = tmp[['Close','Low']].min(axis=1)
    tmp['downturn'] = 0
    tmp['return'] = 0
    for i in range(0,tmp.shape[0]):
        beg_date = tmp.index[i]
        end_date = beg_date + pd.DateOffset(months=1)
        if end_date in tmp.index:
            ret,downturn,dt = _downturn_helper(tmp.loc[beg_date:end_date])
            tmp['downturn'].loc[dt] = downturn
            tmp['return'].loc[dt] = ret
        else:
            continue
    return tmp[['downturn']].resample('MS').max()

def _downturn_helper(df):
    """
    helper function to label_downturn()

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    tuple of 0 or 1 and date
        0 : no cyclical contraction
        1 : cyclical contraction
        
    """
    for i in range(0,df.shape[0]):
        ret = 100*(df['lowest'][i]/df['Close'][0]-1)       
        if ret < -9.5:
            return (ret,1,df.index[i])
    return (0,0,df.index[i])

def xlbl_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month = label.month_name()[:3]
    if month in ['Jan','Aug']:
        month += f'\n{label.year}'
    return month

def create_pca(fin_ts,macro_ts):
    """
    takes in two dataframes, standardizes them, calculates 
    principal components (PCs) for each dataframe, 
    and returns a merged dataframe of PCs
    """
    # standardize the data
    fin_factors = StandardScaler().fit_transform(fin_ts.dropna().values)
    macro_factors = StandardScaler().fit_transform(macro_ts.dropna().values)
    
    # get PCA class and apply to features
    fin_pca = PCA()
    macro_pca = PCA()
    prinComp = fin_pca.fit_transform(fin_factors)
    prinComp2 = macro_pca.fit_transform(macro_factors)
    
    # 3 principal components from financial dataset
    fin_pc_df = pd.DataFrame(prinComp[:,:3], columns = ['PC1', 'PC2','PC3'])
    fin_pc_df.index = fin_ts.dropna().index
    
    # 5 principal components from macro dataset
    macro_pc_df = pd.DataFrame(prinComp2[:,:5], columns = ['PC1', 'PC2','PC3','PC4','PC5'])
    macro_pc_df.index = macro_ts.dropna().index
    
    # Use all 8 principal components to identify cyclical contraction/expansion
    pc_df = pd.concat([fin_pc_df,macro_pc_df],axis=1)
    pc_df.dropna(inplace=True)
    pc_df.columns = ['PC'+str(i) for i in range(1,pc_df.shape[1]+1)]
    pc_df = pc_df.shift(2)
    return pc_df

def stationary(ts,spd):
    df = ts.copy()
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()
    df['1m'] = df[spd].pct_change()*100    
    adf_1m = adfuller(df['1m'].dropna().values)
    if adf_1m[1] <= 0.05:
        return True
    else:
        return False