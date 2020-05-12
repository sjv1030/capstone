# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:51:41 2020

@author: sjv1030_hp
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import json
import os
from fredapi import Fred
from functions import *
fred = Fred(api_key='[INSERT YOUR API KEY HERE]')

# get historical market data
etfs = ["SPY","QQQ","DIA","EEM","IWM","IWO","IWF",
        "GLD","GDX","SLV",
        "TLT","LQD","HYG","MBB","MUB",
        "XLF","XBI","XLV","XRT","XLP","XLY","XLU","XLK","XLC","XLE","XOP",
        "OIH","XME","ITB","KRE","SMH","XLI"]

indices = ["^RUT","^GSPC","^DJI","^NDX"]

etf_data = yf.download(
        tickers = etfs,
        period = "15y",
        interval = "1d",
        group_by = 'Ticker',
        # adjust all OHLC automatically
        auto_adjust = True)

mkts = yf.download(
        tickers = indices,
        period = "20y",
        interval = "1d",
        group_by = 'Ticker',
        # adjust all OHLC automatically
        auto_adjust = True)

# save market prices to database
os.chdir('../data')
prices = sqlite3.connect('mkt_prices.db')
for col in etf_data.columns.get_level_values(0).drop_duplicates().values:
    etf_data[col].to_sql(col,prices,if_exists='replace')
    
for col in mkts.columns.get_level_values(0).drop_duplicates().values:
    if col == '^RUT':
        key = 'RTY' # Russell 2000
    elif col == '^GSPC':
        key = 'SP500' # S&P 500
    elif col == '^DJI':
        key = 'DOW' # Dow Jones Industrial Average
    elif col == '^NDX':
        key = 'NDX' # Nasdaq 100
    mkts[col]['Close'].to_sql(key,prices,if_exists='replace')
prices.close()

# confirm tables were created for each ETF
prices = sqlite3.connect('mkt_prices.db')
cursor = prices.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# print a sample of prices for an ETF
spy = pd.read_sql_query("SELECT * FROM SPY", prices)
print(spy.head())
print(spy.tail())

# close connection
prices.close()

# save macro data to database
macro_dict = dict()

# LABOR MARKET
macro_dict['PAYEMS'] = "US Total Employment - Establishment"
macro_dict['UNRATE'] = "US Unemployment Rate"
macro_dict['AHETPI'] = "US Avg. Hourly Earnings of Production and Nonsupervisory Employees"
macro_dict['EMRATIO'] = "US Employment - Population Ratio"
# macro_dict['ICSA'] = "US Initial Claims"
macro_dict['AWHI'] = "US Aggregate Hours worked of Production and Nonsupervisory Employees"

# HOUSING
macro_dict['TTLCONS'] = "US Total Construction Spending"
macro_dict['PRRESCONS'] = "US Total Construction Spending - Private Residential"
macro_dict['HOUST'] = "US Housing Starts"
macro_dict['HOUST1F'] = "US Housing Starts - Single Family"
macro_dict['UNDCONTSA'] = "US Housing Units Under Construction"
macro_dict['PERMIT'] = "US Building Permits"
macro_dict['PERMIT1'] = "US Building Permits - Single Family"
macro_dict['HSN1F'] = "US New Single Family Houses Sold"
macro_dict['MSPNHSUS'] = "US Median New House Prices NSA"
macro_dict['MORTGAGE30US'] = "US 30-YEAR MTG RATE"
macro_dict['MORTGAGE15US'] = "US 15-YEAR MTG RATE"

# MANUFACTURING
macro_dict['INDPRO'] = "US Industrial Production"
macro_dict['TCU'] = "US Capacity Utilization"
macro_dict['IPMAN'] = "US IP - Manufacturing"
macro_dict['IPG3361T3S'] = "US IP - Auto Manufacturing"
macro_dict['IPBUSEQ'] = "US IP - Business Eqp"
macro_dict['GACDISA066MSFRBNY'] = "Empire Survey - current index"
macro_dict['GAFDISA066MSFRBNY'] = "Empire Survey - future index"
macro_dict['GACDFSA066MSFRBPHI'] = "Philly Fed Survey - current index"
macro_dict['GAFDFSA066MSFRBPHI'] = "Philly Fed Survey - future index"

# Consumer
macro_dict['RRSFS'] = "US Real Retail Sales"
macro_dict['RSAFS'] = "US Retail Sales"
macro_dict['MARTSSM44W72USS'] = "US Retail Sales ex Auto and Parts and Gasoline"
macro_dict['RSBMGESD'] = "US Building Mats Sales"
macro_dict['RSDBS'] = "US Food Sales"
macro_dict['UMCSENT'] = "U Mich Sentiment"

# MACRO
macro_dict['GEPUCURRENT'] = "Global Policy Uncertainty Index"
macro_dict['USEPUINDXD'] = "US Policy Uncertainty Index"
macro_dict['WLEMUINDXD'] = "US Market Economic Uncertainty Index"
macro_dict['NFCI'] = "Chicago Fed Financial Conditions Index"

# INFLATION
macro_dict['CPIAUCSL'] = "US CPI"
macro_dict['CPILFESL'] = "US Core CPI"
macro_dict['PCEPI'] = "US PCE"
macro_dict['PCEPILFE'] = "US Core PCE"
macro_dict['T5YIFR'] = "US 5-year, 5-year Forward Inflation Expectations"
macro_dict['T5YIE'] = "US 5-year Forward Inflation Expectations"

# Financial
macro_dict['T10Y2Y'] = "US Yield Curve - 10s2s"
macro_dict['T10Y3M'] = "US Yield Curve - 10s3M"
macro_dict['TEDRATE'] = "US TED Spread"
macro_dict['BAA10Y'] = "Moodys BBB Spread"
macro_dict['AAA10Y'] = "Moodys AAA Spread"
macro_dict['CPFF'] = "3M Commerical Paper less Fed Funds Rate"
macro_dict['DGS5'] = "US 5-Year Constant Maturity Rate"
macro_dict['DGS30'] = "US 30-Year Constant Maturity Rate"
macro_dict['DTWEXBGS'] = "US Trade-Weighted Dollar"
macro_dict['DTWEXEMEGS'] = "US Trade-Weighted Dollar - EMs"
macro_dict['VIXCLS'] = "VIX"
macro_dict['US530'] = "US Yield Curve - 10s2s" # this series will be created below

fin_tickers = ['T10Y2Y','T10Y3M','TEDRATE','BAA10Y','AAA10Y','CPFF','DGS5',
               'DGS30','DTWEXBGS','DTWEXEMEGS','VIXCLS','T5YIE','T5YIFR',
               'MORTGAGE30US','MORTGAGE15US']

dates_list = pd.date_range(start='2000-01-01', periods=244, freq='MS')

# open connection
os.chdir('../data')
macro_con = sqlite3.connect('macro.db')
for tk in macro_dict.keys():
    if tk in ['US530']: continue
    if tk in fin_tickers:
        tmp = fred.get_series(tk)
        tmp.to_sql(tk,macro_con,if_exists='replace')
    else:
        tmp = fred.get_series_all_releases(tk)
        tmp.to_sql(tk,macro_con,if_exists='replace')
# close connection
macro_con.close()

# create a US yieldcurve using US 30-year yield and US 5-year yield
dgs30 = get_table('DGS30','macro')
dgs30.columns = ['date','US30']
os.chdir('../data')
dgs30['date'] = pd.to_datetime(dgs30['date'])
dgs30.set_index('date',inplace=True)
dgs30['US30'].fillna(method='ffill')

dgs5 = get_table('DGS5','macro')
os.chdir('../data')
dgs5.columns = ['date','US5']
dgs5['date'] = pd.to_datetime(dgs5['date'])
dgs5.set_index('date',inplace=True)
dgs5['US5'].fillna(method='ffill')

usyc = pd.concat([dgs30,dgs5],axis=1)
usyc['US530'] = (usyc['US30']-usyc['US5'])*100

# store the US 30-5 yield curve
macro_con = sqlite3.connect('macro.db')
usyc['US530'].to_sql('US530',macro_con,if_exists='replace')
macro_con.close()

# confirm tables were created
macro_con = sqlite3.connect('macro.db')
cursor = macro_con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# print a sample
payems = pd.read_sql_query("SELECT * FROM PAYEMS", macro_con)
print(payems.head())
print(payems.tail())

# close connection
macro_con.close()

# save macro dictionary
with open('macro_dict.json', 'w') as f:
    json.dump(macro_dict, f)
