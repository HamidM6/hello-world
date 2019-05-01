"""Bayesian dynamic linear model"""

import pydlm
import apollo.src.model_hyperparameter_tuning

def fit_bdlm (
                data,
                time_grain,
                model_time_series_required_length,
                training_endog,
                training_exog_var,
                training_length,
                holdout_end,
                holdout_exog_var,
                complete_exog_var,
                complete_endog,
                complete_length,
                complete_end,
                forecast_exog_var,
                error_logger
              ):
        """

        :param data:
        :param time_grain:
        :param model_time_series_required_length:
        :param training_endog:
        :param training_exog_var:
        :param training_length:
        :param holdout_end:
        :param holdout_exog_var:
        :param complete_exog_var:
        :param complete_endog:
        :param complete_length:
        :param complete_end:
        :param forecast_exog_var:
        :param error_logger:
        :return:
        """
        model = 'bdlm'    
    
        if data['training_length_in_years'].sample(1).values.item(0) >= model_time_series_required_length.get(model, 0.5) and data['time_series_class'].sample(1).values.item(0) == 'nominal':
                                            
                try:
                    
                    discount = 0.96
                    bdlm_training_model = pydlm.dlm(training_endog)
                    linear_trend = pydlm.trend(degree=0, discount=discount, name='trend')
                    if time_grain == 'week':
                        period = 52
                    elif time_grain == 'month':
                        period = 12
                    seasonality = pydlm.seasonality(period=period, discount=discount, name='month')
#                    dynamic = pydlm.dynamic(features=training_exog_var.values.tolist(), discount=discount, name='event_calendar')
                    bdlm_training_model = bdlm_training_model + linear_trend + seasonality #+ dynamic
                    bdlm_training_model = src.model_hyperparameter_tuning.tune_dlm(bdlm_training_model)
                    bdlm_training_model.fit()
                    
                    (
                        bdlm_training_fittedvalues,
                        bdlm_training_fittedvariance
                    ) = bdlm_training_model.predictN(
                                                        N = training_length,
                                                        date = 1
#                                                        , featureDict = {'event_calendar': training_exog_var.values.tolist()}
                                                    )
                    
                    (
                        bdlm_holdout_prediction,
                        bdlm_holdout_variance
                    ) = bdlm_training_model.predictN(
                                                        N = holdout_end - training_length + 1,
                                                        date = bdlm_training_model.n - 1
#                                                        , featureDict = {'event_calendar': holdout_exog_var.values.tolist()}
                                                    )
                    
                    bdlm_forecast_model = pydlm.dlm(complete_endog)
#                    dynamic = pydlm.dynamic(features=complete_exog_var.values.tolist(), discount=discount, name='event_calendar')
                    bdlm_forecast_model = bdlm_forecast_model + linear_trend + seasonality #+ dynamic
                    bdlm_forecast_model = src.model_hyperparameter_tuning.tune_dlm(bdlm_forecast_model)
                    bdlm_forecast_model.fit()
                    
                    (
                        bdlm_forecast,
                        bdlm_forecast_variance
                    ) = bdlm_forecast_model.predictN(
                                                        N = complete_end - complete_length + 1,
                                                        date = bdlm_forecast_model.n - 1
#                                                        , featureDict = {'event_calendar': forecast_exog_var.values.tolist()}
                                                    )
            
                except Exception as e:
                    
                    bdlm_training_model = None
                    bdlm_training_fittedvalues = None
                    bdlm_holdout_prediction = None
                    bdlm_forecast_model = None
                    bdlm_forecast = None    
                    error_logger.error('error in model fit for ' + model + ' ' + ' '.join(data.name) + ' with error ' + str(e))
        
        else:
            
             bdlm_training_model = None
             bdlm_training_fittedvalues = None
             bdlm_holdout_prediction = None
             bdlm_forecast_model = None
             bdlm_forecast = None         
        
        
        return bdlm_training_model, bdlm_training_fittedvalues, bdlm_holdout_prediction, bdlm_forecast_model, bdlm_forecast
