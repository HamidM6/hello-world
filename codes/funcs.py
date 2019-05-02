"""
.. moduleauthor:: UF31246
"""

import pandas
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _combine_into_df(x):
    
    """list to dataframe"""
    
    result = pandas.DataFrame()
    for i in x:
        result = result.append(i)
    
    return result


def wfa(actual, forecast, epsilon):
    """return wfa """

    abs_error = abs(actual - forecast)
    
    wfa = max(0, 1 - 2*sum(abs_error)/(epsilon + sum(forecast) + sum(actual)))
    
    return wfa

def trend_num_to_str(trend_num):
    """return trend string given a number from param space"""

    if int(round(trend_num)) == 0:
        trend = 'n'
    elif int(round(trend_num)) == 1:
        trend = 'c'
    elif int(round(trend_num)) == 2:
        trend = 't'
    else:
        trend = 'ct'
        
    return trend


def sarimax_predict(
        
        training_endog_var,
        holdout_endog_var,
        exog_var,
        training_length,
        holdout_length,
        holdout_exog,
        order,
        seasonal_order,
        trend
        
        ):
    """sarimax predict"""
    
    try:
        
        sarimax_fit = SARIMAX(endog = training_endog_var,
                              exog = exog_var,
                              order = order,
                              seasonal_order = seasonal_order,
                              trend = trend
                              ).fit()
        
    except Exception as e:
        
        if 'maxlag' in str(e):
            
            try:
                
                sarimax_fit = SARIMAX(endog = training_endog_var,
                                      exog = exog_var,
                                      order = order,
                                      seasonal_order = (0,0,0,0),
                                      trend = trend
                                      ).fit()
                
            except Exception as e:
                print(e)
                sarimax_fit = None
    
        elif 'enforce_invertibility' in str(e):
            
            try:
                sarimax_fit = SARIMAX(endog = training_endog_var,
                                      exog = exog_var,
                                      order = order,
                                      seasonal_order = seasonal_order,
                                      trend = trend
                                      ).fit(start_params=[0, 0, 0, 1])
                
            except Exception as e:
                print(e)
                sarimax_fit = None

        else:
            sarimax_fit = None
            
    if sarimax_fit is not None:
        
        try:
            
            y_pred = sarimax_fit.predict(start = training_length,
                                 end = training_length + holdout_length - 1,
                                 exog = holdout_exog
                                 )
        except:
            y_pred = None

    else:
        y_pred = None
        
    return (holdout_endog_var, y_pred)

