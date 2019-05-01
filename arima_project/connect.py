# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:40:51 2019

@author: UF31246
"""
import os
os.chdir('C:/Users/UF31246/Desktop/to-do/ts_clustering')

import pandas
import numpy
import pmdarima as pm
pandas.set_option('display.expand_frame_repr',False)

filled_fact = pandas.read_pickle('filled_fact.pkl').reset_index(drop=True)

key = ['retailer', 'material']
response = 'order_unit'
time_grain = 'month'
epsilon = 1e-7

# forecast and retrun wfa

def wfa(actual, forecast, epsilon):
    
    abs_error = abs(actual - forecast)
    
    wfa = max(0, 1 - 2*sum(abs_error)/(epsilon + sum(forecast) + sum(actual)))
    
    return wfa


def _arima_wfa(x):
    
    subset = x.copy()
    train = subset[subset.data_split == 'Training']
    test = subset[subset.data_split == 'Holdout']
    
    train_y = train[response].values
    
    # fit stepwise auto-ARIMA
    try:
        arima_fit = pm.auto_arima(train_y, start_p=0, start_q=0,
                                     max_p=3, max_q=3, m=12,
                                     start_P=0, seasonal=True,
                                     d=1, D=1, trace=True,
                                     error_action='ignore',  # don't want to know if an order does not work
                                     suppress_warnings=True,  # don't want convergence warnings
                                     stepwise=True)  # set to stepwise
        
        y_pred = arima_fit.predict(n_periods=len(test))
        y_true = test[response].values
        
        subset_wfa = wfa(y_true, y_pred, epsilon)
        
    except:
        subset_wfa = 0

    return subset_wfa

# compute
wfa_dict = {}
gb = filled_fact.groupby(key)

for gbkey in gb.groups:
    
    subset = gb.get_group(gbkey)
    wfa_dict[gbkey] = _arima_wfa(subset)
    
