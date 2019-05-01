"""Exponentially Weighted Moving Average"""

import numpy
import pandas

def fit_ewm(
                    data_name,
                    model_time_series_required_length,
                    input_endog,
                    input_dates,
                    input_length,
                    forecast_length,
                    time_grain,
                    input_endog_shifted,
                    forecast_shifted_response,
                    error_logger,
                    training_length_in_years,
                    time_series_class,
                    holidays,
                    training_exog_var,
                    forecast_exog_var
              ):
        """

        :param data_name:
        :param model_time_series_required_length:
        :param input_endog:
        :param input_dates:
        :param input_length:
        :param forecast_length:
        :param time_grain:
        :param input_endog_shifted:
        :param forecast_shifted_response:
        :param error_logger:
        :param training_length_in_years:
        :param time_series_class:
        :param holidays:
        :param training_exog_var:
        :param forecast_exog_var:
        :return:
        """
        model = 'ewm_model'
        
        if time_series_class == 'near_disco':
                
                if time_grain == 'month':
                    span = 2
                else:
                    span = 4
                try:
                    
                    ewm_model = input_endog.ewm(span=span)
                    ewm_fittedvalues = numpy.round(ewm_model.mean())
                    
                    ewm_forecast = numpy.empty(forecast_length)
                    ewm_forecast.fill(ewm_fittedvalues.iloc[-1])
                    ewm_forecast = pandas.Series(ewm_forecast)

                except Exception as e:
                    
                    ewm_model = None
                    ewm_fittedvalues = None
                    ewm_forecast = None
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            ewm_model = None
            ewm_fittedvalues = None
            ewm_forecast = None        
        
        
        return ewm_model, ewm_fittedvalues, ewm_forecast
