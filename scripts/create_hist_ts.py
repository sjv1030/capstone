# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:42:50 2020

@author: sjv1030_hp
"""

import os
import json
from functions import *

fin_tickers = ['T10Y2Y','T10Y3M','TEDRATE','BAA10Y','AAA10Y','CPFF','DGS5',
               'DGS30','DTWEXBGS','DTWEXEMEGS','VIXCLS','T5YIE','T5YIFR',
               'MORTGAGE30US','MORTGAGE15US','US530']

# get macro data
tbl_list = print_tables('macro')

# get macro dictionary
os.chdir('../data')
with open('macro_dict.json') as f:
    macro_dict2 = json.load(f)
os.chdir('../scripts')

macro_factors = dict()

# store PIT macro data in a dictionary
for _ in tbl_list: 
    if _ not in ['USEPUINDXD','WLEMUINDXD']: continue
    if _ in fin_tickers:
        tmp = get_table(_,'macro').fillna(method='ffill')
        tmp.columns = ['date',_]
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp.set_index('date',inplace=True)
        tmp_m = tmp.resample('MS').last()
        tmp_m['ratio6ma'] = tmp_m[_]/tmp_m[_].rolling(6).mean()
        if _ in ['DTWEXBGS','DTWEXEMEGS','VIXCLS']:
            tmp_m['mom'] = tmp_m[_].pct_change()*100
            tmp_m['yoy'] = tmp_m[_].pct_change(12)*100
            tmp_m['yoy6ma'] = tmp_m['yoy'].rolling(6).mean()
        else:
            tmp_m['mom'] = tmp_m[_].diff()*100
            tmp_m['yoy'] = tmp_m[_].diff(12)*100
            tmp_m['yoy6ma'] = tmp_m['yoy'].rolling(6).mean()
        macro_factors[_] = tmp_m
    else:
        if _ in ['NFCI','USEPUINDXD','WLEMUINDXD']:
            tmp = get_table(_,'macro')
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp.set_index('date',inplace=True)
            tmp = tmp.resample('MS').last()
            tmp.reset_index(inplace=True)
            tmp = create_ts(tmp,td='2020-05-01')
        else: 
           tmp = create_ts(get_table(_,'macro'),td='2020-05-01')           
        tmp['ratio6ma'] = tmp['value']/tmp['value'].rolling(6).mean()
        tmp['yoy'].interpolate(method='linear',inplace=True)
        tmp['yoy6ma'] = tmp['yoy'].rolling(6).mean()
        macro_factors[_] = tmp

# save macro factors in a table
os.chdir('../data')
# open connection
factors = sqlite3.connect('factors.db')
for tk in macro_factors.keys():
    macro_factors[tk].to_sql(tk,factors,if_exists='replace')
# close connection
factors.close()