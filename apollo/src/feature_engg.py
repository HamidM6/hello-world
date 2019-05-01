"""
# automated feature creation
"""
import os
import pandas
import numpy as np
import ruptures as rpt
from luminol.anomaly_detector import AnomalyDetector
import pmdarima as pm

def detect_level_shift(time_series_data_frame, key, response, time_grain):
    """

    :param time_series_data_frame:
    :param key:
    :param response:
    :param time_grain:
    :return:
    """
    column0, column1 = key[0], key[1]
    
    level_shift_data_frame = pandas.DataFrame(columns=[column0, column1, time_grain])
    gb = time_series_data_frame.groupby(key)
    
    for gkey in gb.groups:
        
        subset = gb.get_group(gkey)

        mode_ratio = np.round(subset[subset[response] == subset[response].mode()[0]].shape[0] / subset.shape[0], 2)
        
        if (subset.shape[0] >= 5) & (mode_ratio < 0.4):
            
            signal = subset[response].values.reshape(subset.shape[0], 1)
            algo = rpt.Pelt(model='rbf').fit(signal)
            result = algo.predict(pen=10)
            if len(result) > 1:
                level_shift_data_frame = level_shift_data_frame.append({column0:gkey[0], column1:gkey[1], time_grain:subset[time_grain].iloc[result[-2]]}, ignore_index=True)
        
    return level_shift_data_frame


def detect_anomaly(time_series_data_frame, key, time_grain):
    """

    :param time_series_data_frame:
    :param key:
    :param time_grain:
    :return:
    """
    col0, col1 = key[0], key[1]
    
    anomaly_data_frame = pandas.DataFrame(columns=[col0, col1, time_grain])
    
    gb = time_series_data_frame.groupby(key)
    
    for gkey in gb.groups:
        
        subset= gb.get_group(gkey).sort_values(time_grain).reset_index(drop=True)
        
        if subset.shape[0] >=5:
            
            ts = subset.pos_unit.to_dict()
            my_detector = AnomalyDetector(ts, score_threshold=9.25)
            anomalies = my_detector.get_anomalies()
            
            if anomalies:
                
                anom_week = subset[time_grain][anomalies[0].get_time_window()[0]]
                anomaly_data_frame = anomaly_data_frame.append({col0:gkey[0], col1:gkey[1], time_grain:anom_week}, ignore_index=True)

    return anomaly_data_frame        


def detect_arima_orders(time_series_data_frame, key, response, time_grain):
    """

    :param time_series_data_frame:
    :param key:
    :param response:
    :param time_grain:
    :return:
    """
    col0, col1 = key[0], key[1]
    arima_order_data_frame = pandas.DataFrame(columns=[col0, col1, 'train_ar', 'train_d', 'train_ma', 'complete_ar', 'complete_d', 'complete_ma'])
    
    gb = time_series_data_frame.groupby(key)
    
    for gbkey in gb.groups:
        
        subset = gb.get_group(gbkey).sort_values(time_grain)
        training_endog = subset[subset.data_split == 'Training'][response].reset_index(drop=True)
        complete_endog = subset[subset.data_split != 'Forecast'][response].reset_index(drop=True)

        try:
    
            aa_train = pm.auto_arima(y=training_endog, start_p=0, start_q=0, m=52, stepwise=True)
            aa_complete = pm.auto_arima(y=complete_endog, start_p=0, start_q=0, m=52, stepwise=True)
        
            arima_order_data_frame = arima_order_data_frame.append({col0:gbkey[0], col1:gbkey[1], 'train_ar':aa_train.order[0], 'train_d':aa_train.order[1], 'train_ma':aa_train.order[2], 'complete_ar':aa_complete.order[0], 'complete_d':aa_complete.order[1], 'complete_ma':aa_complete.order[2]}, ignore_index=True)
            
        except Exception as e:
            
            print(e)
            #error_logger.error('error in detect_arima_order for ' + gbkey + ' with error ' + str(e))
        
    return arima_order_data_frame
