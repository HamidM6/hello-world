"""future values are predicted to be zero"""
import numpy
import pandas 

def fit_zero(
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
        model = 'zero_model'    
    
        if time_series_class == 'is_disco':
                    
                try:
                    
                    zero_model = 0
                    
                    zero_model_fittedvalues = numpy.empty(input_length)
                    zero_model_fittedvalues.fill(zero_model)
                    zero_model_fittedvalues = pandas.Series(zero_model_fittedvalues)
                    
                    zero_model_forecast = numpy.empty(forecast_length)
                    zero_model_forecast.fill(zero_model)
                    zero_model_forecast = pandas.Series(zero_model_forecast)
            
                except Exception as e:
                    zero_model = None
                    zero_model_fittedvalues = None
                    zero_model_forecast = None    
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            zero_model = None
            zero_model_fittedvalues = None
            zero_model_forecast = None        
        
        
        return zero_model, zero_model_fittedvalues, zero_model_forecast