"""
# apollo demand forecast platform
# functions to determine attributes related to time series
# @author: vikram govindan
"""
import pandas
import numpy

def fill_in_missing_periods(
                                time_series_data_frame,
                                key,
                                time_grain, 
                                response_dict,
                                pass_through_key_attributes
                           ):
    """

    :param time_series_data_frame:
    :param key:
    :param time_grain:
    :param response_dict:
    :param pass_through_key_attributes:
    :return:
    """
    key_with_attributes = key + pass_through_key_attributes

    if time_grain == 'week':
        filled_in_time_series_data_frame = time_series_data_frame.set_index(time_grain).groupby(key_with_attributes).resample('W-SAT').mean()       
    elif time_grain == 'month':
         filled_in_time_series_data_frame = time_series_data_frame.set_index(time_grain).groupby(key_with_attributes).resample('M', convention = 'start').mean()
    
    for k, response in response_dict.items():
        filled_in_time_series_data_frame[response] = filled_in_time_series_data_frame[response].fillna(0)
    filled_in_time_series_data_frame['parallel_partition_dense_rank'] = filled_in_time_series_data_frame['parallel_partition_dense_rank'].fillna(method = 'backfill')
    
    return filled_in_time_series_data_frame


def pivot_and_fill_in_missing_periods(
                                        event_calendar,
                                        time_grain,
                                        event_calendar_key
                                    ):
    """

    :param event_calendar:
    :param time_grain:
    :param event_calendar_key:
    :return:
    """
    event_calendar['event_flag'] = 1
    pivot_event_calendar = pandas.pivot_table(
                                                  data = event_calendar,
                                                  columns = 'event_name',
                                                  aggfunc = numpy.mean,
                                                  index = event_calendar_key,
                                                  values = 'event_flag',
                                                  fill_value = 0
                                              ).reset_index()
    
    if time_grain == 'week':        
        pivot_event_calendar = pivot_event_calendar.set_index(time_grain).groupby('division').resample('W-SAT').mean().fillna(0).reset_index()        
    
    return pivot_event_calendar


def machine_cleanse_response(
                                x,
                                training_response,
                                forecast_response,
                                stdevs_to_cap_outliers
                            ):
    """

    :param x:
    :param training_response:
    :param forecast_response:
    :param stdevs_to_cap_outliers:
    :return:
    """
    training_x = x[x.data_split == 'Training']
    complete_x = x[x.data_split != 'Forecast']    
        
    tr = training_x[training_response]
    training_response_mean = tr.mean()
    training_response_std = tr.std()    
    training_response_ub = numpy.round(training_response_mean + stdevs_to_cap_outliers * training_response_std)
    x[training_response] = x[training_response].clip(upper = training_response_ub)

    cr = complete_x[forecast_response]
    complete_response_mean = cr.mean()
    complete_response_std = cr.std()    
    complete_response_ub = numpy.round(complete_response_mean + stdevs_to_cap_outliers * complete_response_std)
    x[forecast_response] = x[forecast_response].clip(upper = complete_response_ub)
    
    return x