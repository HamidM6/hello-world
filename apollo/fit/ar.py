# apollo demand forecast platform
# model orchestration
# @author: vikram govindan

import statsmodels.api 

def fit_ar(
            data,
            model_time_series_required_length,
            training_endog,
            training_dates,
            training_length,
            holdout_end,
            complete_endog,
            complete_dates,
            complete_length,
            complete_end,
            error_logger
          ):
        
        
        model = 'ar'
    
        if data['training_length_in_years'].sample(1).values.item(0) < model_time_series_required_length.get(model, 0.5):
        
                try: 
                    
                    ar_training_model = statsmodels.api.tsa.AR(
                                                                    endog = training_endog,
                                                                    dates = training_dates
                                                            ).fit()
                    ar_training_fittedvalues = ar_training_model.fittedvalues
                    ar_holdout_prediction = ar_training_model.predict(
                                                                        start = training_length,
                                                                        end =   holdout_end
                                                                    )
                    
                    ar_forecast_model = statsmodels.api.tsa.AR(
                                                                    endog = complete_endog,
                                                                    dates = complete_dates                                                  
                                                            ).fit()
                    ar_forecast = ar_forecast_model.predict(
                                                            start = complete_length,
                                                            end =   complete_end
                                                        )
                except Exception as e:
                    
                    ar_training_model = None
                    ar_training_fittedvalues = None
                    ar_holdout_prediction = None
                    ar_forecast_model = None
                    ar_forecast = None
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data.name) + ' with error ' + str(e))
        
        else:
        
            ar_training_model = None
            ar_training_fittedvalues = None
            ar_holdout_prediction = None
            ar_forecast_model = None
            ar_forecast = None
        
        
        return ar_training_model, ar_training_fittedvalues, ar_holdout_prediction, ar_forecast_model, ar_forecast