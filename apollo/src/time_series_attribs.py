"""
functions to determine attributes related to time series
"""

import pandas
import numpy

def _time_series_length_conditions(time_series_length, time_grain):
    """

    :param time_series_length:
    :param time_grain:
    :return:
    """
    # segment time series length
    
    if time_grain == 'week':
        return time_series_length/52.00
    elif time_grain == 'month':
        return time_series_length/12.00


def _get_latest_date(x):
    """

    :param x:
    :return:
    """
    y = x[x!=max(x)].nlargest(1)
    
    if y.empty is True:
        return x.min()
    else:
        return y


def _check_is_disco(
                    x,
                    max_date,
                    zero_weeks_to_disco
                    ):
    """

    :param x:
    :param max_date:
    :param zero_weeks_to_disco:
    :return:
    """
    is_disco = (max_date - x[x!=max(x)].nlargest(1)).dt.days/7 >= zero_weeks_to_disco
    if is_disco.empty is True:
        return True
    else:
        return is_disco


def get_time_series_attribs(
                                time_series_data_frame, 
                                key, 
                                time_grain, 
                                training_response, 
                                max_date, 
                                zero_weeks_to_disco, 
                                epsilon,
                                holdout_start_date
                            ):
    """

    :param time_series_data_frame:
    :param key:
    :param time_grain:
    :param training_response:
    :param max_date:
    :param zero_weeks_to_disco:
    :param epsilon:
    :param holdout_start_date:
    :return:
    """
    time_series_attribs = time_series_data_frame[
                                                    time_series_data_frame[time_grain] < max_date
                                                ].groupby(key)[training_response].size().to_frame(name='time_series_length')
    
    time_series_attribs = time_series_attribs.join(
                                                    time_series_data_frame[time_series_data_frame[time_grain] < max_date].groupby(key)[training_response].agg
                                                                                                    (
                                                                                                        {
                                                                                                        'long_term_mean':       lambda x: x.mean(),
                                                                                                        'long_term_min':        lambda x: x.min(),
                                                                                                        'long_term_max':        lambda x: x.max(),
                                                                                                        'long_term_stdev':      lambda x: x.std(),
                                                                                                        'coeff_of_variation':   lambda x: 100*(x.std()/(x.mean() + epsilon))
                                                                                                        }
                                                                                                    )
                                                    )                                                                                 
                                                                                    
    time_series_attribs['time_series_length_in_years'] = time_series_attribs['time_series_length'].apply(
                                                                                                            _time_series_length_conditions, 
                                                                                                            args = (time_grain,)
                                                                                                        )

    time_series_attribs = time_series_attribs.join(
                                                        time_series_data_frame[
                                                                                time_series_data_frame[time_grain] < holdout_start_date
                                                                              ].groupby(key)[training_response].agg
                                                                                                            (
                                                                                                               {
                                                                                                                'training_length_in_years': lambda x: len(x)
                                                                                                               } 
                                                                                                            )
                                                  )
    
    
    if time_grain == 'week':
        time_series_attribs['training_length_in_years'] = time_series_attribs['training_length_in_years']/52.00
    elif time_grain == 'month':
        time_series_attribs['training_length_in_years'] = time_series_attribs['training_length_in_years']/12.00
    
    time_series_attribs = time_series_attribs.join(
                                                    time_series_data_frame.groupby(key)[time_grain].agg
                                                                                                        (
                                                                                                            {
                                                                                                                'earliest_date':    lambda x: x.min(),
                                                                                                                'latest_date':      lambda x: _get_latest_date(x),
                                                                                                                'is_disco':         lambda x: _check_is_disco(x, max_date, zero_weeks_to_disco)
                                                                                                            }
                                                                                                        )
                                                   )
    if time_grain == 'week':
            time_series_attribs['missing_periods'] = numpy.where(
                                                                    time_series_attribs['time_series_length'] > 0,
                                                                    (((pandas.to_timedelta(time_series_attribs['latest_date']) - pandas.to_timedelta(time_series_attribs['earliest_date'])).dt.days)/7 + 1) - time_series_attribs['time_series_length'],
                                                                    None
                                                                )
    elif time_grain == 'month':
            time_series_attribs['missing_periods'] = numpy.where(
                                                                    time_series_attribs['time_series_length'] > 0,
                                                                    (((pandas.to_timedelta(time_series_attribs['latest_date']) - pandas.to_timedelta(time_series_attribs['earliest_date'])).dt.days)/30 + 1) - time_series_attribs['time_series_length'],
                                                                    None
                                                                )

        # explore using describe
        # raw_fact.groupby(key).describe() 
    
    
    time_series_attribs['cov_bucket'] = time_series_attribs['coeff_of_variation'].apply(bucket_coeff_of_variation)
    time_series_attribs['training_length_bucket'] = time_series_attribs['training_length_in_years'].apply(bucket_length_in_years)
    time_series_attribs['time_series_length_bucket'] = time_series_attribs['time_series_length_in_years'].apply(bucket_length_in_years)
    
    return time_series_attribs

def time_series_sample_labeling_conditions(
                                                x, 
                                                max_date, 
                                                holdout_start_date
                                          ):
    """

    :param x:
    :param max_date:
    :param holdout_start_date:
    :return:
    """
    if x > max_date:
        return 'Forecast'
    elif x < holdout_start_date:
        return 'Training'
    else:
        return 'Holdout'


def bucket_coeff_of_variation(x):
    """

    :param x:
    :return:
    """
    if x >= 100:
        return '(1) >= 100%'
    elif 50 <= x < 100:
        return '(2) 50 <= COV < 100'
    elif 25 <= x < 50:
        return '(3) 25 <= COV < 50'
    elif 10 <= x < 25:
        return '(4) 10 <= COV < 25'
    elif 0 <= x < 10:
        return '(5) 0 <= COV < 10'
    else:
        return '(6) Not Defined'


def bucket_length_in_years(x):
    """

    :param x:
    :return:
    """
    if x >= 2:
        return '(1) >= 2'
    elif 1 <= x < 2:
        return '(2) 1 <= #years < 2'
    elif 0.5 <= x < 1:
        return '(3) 0.5 <= #years < 1'
    elif 0.25 <= x < 0.5:
        return '(4) 0.25 <= #years < 0.5'
    else:
        return '(5) < 0.25 years'
