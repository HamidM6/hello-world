"""last year's observations are predicted as future forecasts"""

def fit_last_year(
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
        model = 'last_year'    
    
        if training_length_in_years > model_time_series_required_length.get(model, 1.0):
                                            
                try:
                    
                    last_year = 'response values from previous years'
                    
                    last_year_fittedvalues = input_endog_shifted

                    last_year_forecast = forecast_shifted_response
            
                except Exception as e:
                    last_year = None
                    last_year_fittedvalues = None
                    last_year_forecast = None    
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data_name) + ' with error ' + str(e))
        
        else:
            
            last_year = None
            last_year_fittedvalues = None
            last_year_forecast = None
        
        
        
        return last_year, last_year_fittedvalues, last_year_forecast
