"""future values are predicted to be the average of the past observations"""
import numpy
import pandas 

def fit_mean (
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
        model = 'mean_model'
        
        if training_length_in_years < model_time_series_required_length.get(model, 0.5) or time_series_class == 'not_enough_history':
                                            
                try:
                    
                    mean_model = numpy.round(numpy.mean(input_endog))
                    
                    mean_model_fittedvalues = numpy.empty(input_length)
                    mean_model_fittedvalues.fill(mean_model)
                    mean_model_fittedvalues = pandas.Series(mean_model_fittedvalues)
                    
                    mean_model_forecast = numpy.empty(forecast_length)
                    mean_model_forecast.fill(mean_model)
                    mean_model_forecast = pandas.Series(mean_model_forecast)
                    
            
                except Exception as e:
                    mean_model = None
                    mean_model_fittedvalues = None
                    mean_model_forecast = None
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            mean_model = None
            mean_model_fittedvalues = None
            mean_model_forecast = None
        
        
        
        return mean_model, mean_model_fittedvalues, mean_model_forecast
