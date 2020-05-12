# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:13:08 2020

@author: sjv1030_hp
"""

import os


# step 1 - import functions
os.chdir('../scripts')
import functions

# step 2 - download data
os.chdir('../scripts')
import download_data

# step 3 - create historical time series of 
# macroeconomic and financial data
os.chdir('../scripts')
import create_hist_ts

# step 4 - PCA
os.chdir('../scripts')
import cycle

# step 5 - cluster analysis
os.chdir('../scripts')
import hist_clustering

# step 6 - classification models for ETF spreads
os.chdir('../scripts')
import etf_spreads

# step 7 - portfolio: run a backtest
os.chdir('../scripts')
import portfolio