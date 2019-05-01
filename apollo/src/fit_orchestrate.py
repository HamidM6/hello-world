"""
model orchestration
"""

# parallelize
import dask
from dask.diagnostics import ProgressBar
#from dask.distributed import Client
from apollo.src import fitter
import time
from apollo.src import utils
from apollo import fit
from apollo.src import model_selection


def generate_fit_results(
                            data_dict, 
                            model_list,
                            time_grain,
                            training_response,
                            forecast_response,
                            key,
                            model_fit_error_logger,
                            model_time_series_required_length,
                            epsilon,
                            pass_through_key_attributes,
                            benchmark,
                            holidays,
                            exog_cols,
                            fit_type,
                            ensemble_model_list
                        ):

    
    result = dict((data_dict_key, _fit_orchestrate(  
                                                 data_dict_key,
                                                 data_dict_values['training'] if fit_type == 'training' else data_dict_values['complete'],
                                                 data_dict_values['holdout'] if fit_type == 'training' else data_dict_values['forecast'],
                                                 model_list,
                                                 time_grain,
                                                 training_response,
                                                 forecast_response,
                                                 key,
                                                 model_fit_error_logger,
                                                 model_time_series_required_length,
                                                 epsilon,
                                                 pass_through_key_attributes,
                                                 benchmark,
                                                 holidays,
                                                 exog_cols,
                                                 fit_type,
                                                 data_dict_values['winning_model'] if fit_type == 'complete' else None,
                                                 data_dict_values['best_model'] if fit_type == 'complete' else None,
                                                 ensemble_model_list
                                               
                                               )) for data_dict_key, data_dict_values in data_dict.items())

    return result

def _fit_orchestrate(
                        key_name,
                        training_data,
                        forecast_data,
                        model_list,
                        time_grain,
                        training_response,
                        forecast_response,
                        key,
                        error_logger,
                        model_time_series_required_length,
                        epsilon,
                        pass_through_key_attributes,
                        benchmark,
                        holidays,
                        exog_cols,
                        fit_type,
                        winning_model,
                        best_model,
                        ensemble_model_list
                    ):
        """

        :param key_name:
        :param training_data:
        :param forecast_data:
        :param model_list:
        :param time_grain:
        :param training_response:
        :param forecast_response:
        :param key:
        :param error_logger:
        :param model_time_series_required_length:
        :param epsilon:
        :param pass_through_key_attributes:
        :param benchmark:
        :param holidays:
        :param exog_cols:
        :param fit_type:
        :param winning_model:
        :param best_model:
        :param ensemble_model_list:
        :return:
        """
        training_endog = training_data[training_response]
        training_dates = training_data[time_grain]
        training_length = len(training_data)
        training_exog_data = training_data[exog_cols]
        
        forecast_length = len(forecast_data)
        forecast_exog_data = forecast_data[exog_cols]

        training_length_in_years = training_data['training_length_in_years'].sample(1).values.item(0)
        time_series_class = training_data['time_series_class'].sample(1).values.item(0)
        
        training_shifted_response = training_data['shifted_' + training_response]
        forecast_shifted_response = forecast_data['shifted_' + forecast_response]

        suffix = utils.prediction_suffix_by_fit_type(fit_type)
        fit_results = {}
        fit_results['training_response'] = training_data[training_response]
        fit_results['forecast_response'] = forecast_data[forecast_response]
        fit_results['benchmark' + suffix] = forecast_data['benchmark']
        fit_results['benchmark_fitted_' + fit_type] = training_data['benchmark']
        
        if fit_type == 'complete':
            fit_results['winning_model'] = winning_model
            fit_results['best_model'] = best_model
            if best_model != 'ensemble':
                if winning_model == best_model:
                    m = [winning_model]
                else:
                    m = [best_model, winning_model]
            else:
                if winning_model == best_model:
                    m = ensemble_model_list
                else:
                    m = [winning_model] + ensemble_model_list
        else:
            m = model_list
        
        for model in m:
            
            t_fit_start = time.time()            
            
            prefix = model
                
            if fit_type == 'complete':
                if model == best_model:
                    prefix = 'best_model'
            
            if model != 'benchmark':
                
                f = fitter.fitter(model)
                    
                (
                        
                    fit_results[prefix + '_' + fit_type + '_modelobj'],
                    fit_results[prefix + '_fitted_' + fit_type], 
                    fit_results[prefix + suffix]
                    
                ) = f.fit (
                                data_name = key_name,
                                model_time_series_required_length = model_time_series_required_length,
                                input_endog = training_endog,
                                input_dates = training_dates,
                                input_length = training_length,
                                forecast_length = forecast_length,
                                time_grain = time_grain,
                                input_endog_shifted = training_shifted_response,
                                forecast_shifted_response = forecast_shifted_response,
                                error_logger = error_logger,
                                training_length_in_years = training_length_in_years,
                                time_series_class = time_series_class,
                                holidays = holidays,
                                training_exog_var = training_exog_data,
                                forecast_exog_var = forecast_exog_data
                          )
                    
                t_fit_end = time.time()
                fit_results[model + '_runtime_' + fit_type] = t_fit_end - t_fit_start
                
                if fit_type == 'complete' and best_model != 'ensemble' and best_model == winning_model:
                    fit_results['winning_model' + '_' + fit_type + '_modelobj'] = fit_results[prefix + '_' + fit_type + '_modelobj']
                    fit_results['winning_model' + '_fitted_' + fit_type] =  fit_results[prefix + '_fitted_' + fit_type] 
                    fit_results['winning_model' + suffix] = fit_results[prefix + suffix]                    
                
            else:
                
                fit_results['winning_model' + '_' + fit_type] = 'benchmark'
                fit_results['winning_model' + '_fitted_' + fit_type] =  training_data['benchmark']
                fit_results['winning_model' + suffix] = forecast_data['benchmark']                    

        return fit_results


def non_parallel_model_fit(
                             main_info_logger,
                             data_dict,
                             model_list,
                             time_grain,
                             training_response,
                             forecast_response,
                             key,
                             model_fit_error_logger,
                             model_time_series_required_length,
                             epsilon,
                             pass_through_key_attributes,
                             benchmark,
                             holidays,
                             exog_cols,
                             fit_type,
                             n_keys,
                             ensemble_model_list = None
                         ):
        """

        :param main_info_logger:
        :param data_dict:
        :param model_list:
        :param time_grain:
        :param training_response:
        :param forecast_response:
        :param key:
        :param model_fit_error_logger:
        :param model_time_series_required_length:
        :param epsilon:
        :param pass_through_key_attributes:
        :param benchmark:
        :param holidays:
        :param exog_cols:
        :param fit_type:
        :param n_keys:
        :param ensemble_model_list:
        :return:
        """
        t_training_fit_start = time.time()
        model_type = 'forecast' if fit_type == 'complete' else fit_type
        main_info_logger.info('staring to fit '+ model_type +' models for ' + n_keys + ' groups')        
        
        fit_results_training = generate_fit_results(
                                                                         data_dict = data_dict,
                                                                         model_list = model_list,
                                                                         time_grain = time_grain,
                                                                         training_response = training_response,
                                                                         forecast_response = forecast_response,
                                                                         key = key,
                                                                         model_fit_error_logger = model_fit_error_logger,
                                                                         model_time_series_required_length = model_time_series_required_length,
                                                                         epsilon = epsilon,
                                                                         pass_through_key_attributes = pass_through_key_attributes,
                                                                         benchmark = benchmark,
                                                                         holidays = holidays,
                                                                         exog_cols = exog_cols,
                                                                         fit_type = fit_type,
                                                                         ensemble_model_list = ensemble_model_list
                                                                       )
        
        t_training_fit_end = time.time()
        main_info_logger.info('non parallel '+ model_type +' fit runtime = ' + str((t_training_fit_end - t_training_fit_start)/60) + ' minutes')
        
        return fit_results_training
    

def non_parallel_ensemble_fit(
                                main_info_logger,
                                n_keys,
                                fit_results_dict,
                                ensemble_model_list,
                                fit_type,
                                ensemble
                             ):
    """

    :param main_info_logger:
    :param n_keys:
    :param fit_results_dict:
    :param ensemble_model_list:
    :param fit_type:
    :param ensemble:
    :return:
    """
    t_ensemble_training_fit_start = time.time()
    model_type = 'forecast' if fit_type == 'complete' else fit_type
    main_info_logger.info('staring to fit ensemble '+ model_type +' models for ' + n_keys + ' groups') 
    
    if ensemble:        
        fit_results_dict = fit.ensemble.fit_ensemble(
                                                            fit_results = fit_results_dict,
                                                            ensemble_model_list = ensemble_model_list,
                                                            fit_type = fit_type
                                                        )   
    
    t_ensemble_training_fit_end = time.time()
    main_info_logger.info('non parallel ensemble '+ model_type +' fit runtime = ' + str((t_ensemble_training_fit_end - t_ensemble_training_fit_start)/60) + ' minutes')
    
    return fit_results_dict


def non_parallel_pick_and_pass_winning_and_best_model(
                                                        fit_results_training,
                                                        model_list,
                                                        ensemble,
                                                        post_process_execution_path,
                                                        main_info_logger,
                                                        n_keys,
                                                        epsilon,
                                                        pre_processed_dict
                                                     ):
    """

    :param fit_results_training:
    :param model_list:
    :param ensemble:
    :param post_process_execution_path:
    :param main_info_logger:
    :param n_keys:
    :param epsilon:
    :param pre_processed_dict:
    :return:
    """
    t_model_selection_start = time.time()
    main_info_logger.info('staring model selection ' + n_keys + ' groups')
    
    fit_results_training = model_selection.pick_winning_model(
                                                                    fit_results_training,
                                                                    model_list + ['ensemble'] if ensemble is True else model_list,
                                                                    epsilon
                                                                 )        
        
    pre_processed_dict = model_selection.pass_winning_and_best_model(
                                                                             pre_processed_dict, 
                                                                             fit_results_training
                                                                        )
    
    t_model_selection_end = time.time()
    main_info_logger.info('non parallel model selection runtime = ' + str((t_model_selection_end - t_model_selection_start)/60) + ' minutes')
    
    return (pre_processed_dict, fit_results_training)



def parallel_model_fit(
                            root,
                            run_id,
                            pre_processed_dict,
                            parallel_partitions,
                            model_list,
                            time_grain,
                            training_response,
                            forecast_response,
                            key,
                            model_fit_error_logger,
                            model_time_series_required_length,
                            epsilon,
                            pass_through_key_attributes,
                            benchmark,
                            holidays,
                            exog_cols,
                            fit_type,
                            main_info_logger,
                            n_keys,
                            ensemble_model_list = None
                     ):
    """

    :param root:
    :param run_id:
    :param pre_processed_dict:
    :param parallel_partitions:
    :param model_list:
    :param time_grain:
    :param training_response:
    :param forecast_response:
    :param key:
    :param model_fit_error_logger:
    :param model_time_series_required_length:
    :param epsilon:
    :param pass_through_key_attributes:
    :param benchmark:
    :param holidays:
    :param exog_cols:
    :param fit_type:
    :param main_info_logger:
    :param n_keys:
    :param ensemble_model_list:
    :return:
    """
    t_training_fit_start = time.time()
    model_type = 'forecast' if fit_type == 'complete' else fit_type
        
    dask.config.set(scheduler='processes')
    
    # parallel_compute_graph_filename = root + '\\log\\' + str(run_id) + '_dask_compute_graph_' + fit_type + '.png'
    
    pre_processed_data_set_list = utils.chunk_dict(pre_processed_dict, parallel_partitions)
            
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        fit_task = dask.delayed(generate_fit_results)(
                                                                             data_dict = pre_processed_data_set_list[i],
                                                                             model_list = model_list,
                                                                             time_grain = time_grain,
                                                                             training_response = training_response,
                                                                             forecast_response = forecast_response,
                                                                             key = key,
                                                                             model_fit_error_logger = model_fit_error_logger,
                                                                             model_time_series_required_length = model_time_series_required_length,
                                                                             epsilon = epsilon,
                                                                             pass_through_key_attributes = pass_through_key_attributes,
                                                                             benchmark = benchmark,
                                                                             holidays = holidays,
                                                                             exog_cols = exog_cols,
                                                                             fit_type = fit_type,
                                                                             ensemble_model_list = ensemble_model_list
                                                                         )
        task_list.append(fit_task)
    
    fit_task = dask.delayed(utils.combine_into_dict)(task_list)
    # fit_task.visualize(filename = parallel_compute_graph_filename)
    
    main_info_logger.info('staring to fit '+ model_type +' models for ' + n_keys + ' groups')        
    #client = Client()  # start distributed scheduler locally.  Launch dashboard
    with ProgressBar():
        fit_results = fit_task.compute()
    
    t_training_fit_end = time.time()
    main_info_logger.info('parallel '+ model_type +' fit runtime = ' + str((t_training_fit_end - t_training_fit_start)/60) + ' minutes')
    
    return fit_results


def parallel_fit_ensemble(
                            root,
                            run_id,
                            fit_results_dict,
                            fit_type,
                            parallel_partitions,
                            ensemble_model_list,
                            main_info_logger,
                            n_keys
                        ):
    """

    :param root:
    :param run_id:
    :param fit_results_dict:
    :param fit_type:
    :param parallel_partitions:
    :param ensemble_model_list:
    :param main_info_logger:
    :param n_keys:
    :return:
    """
    t_ensemble_training_fit_start = time.time()
    model_type = 'forecast' if fit_type == 'complete' else fit_type
        
    # parallel_compute_graph_filename = root + '\\log\\' + str(run_id) + '_' + 'dask_compute_graph_'  + '_' + fit_type + '_ensemble.png'
    
    fit_results_dict_list = utils.chunk_dict(fit_results_dict, parallel_partitions)
            
    task_list = []
    for i in range(len(fit_results_dict_list)):
        fit_task = dask.delayed(fit.ensemble.fit_ensemble)(
                                                                fit_results = fit_results_dict_list[i],
                                                                ensemble_model_list = ensemble_model_list,
                                                                fit_type = fit_type
                                                              )
        task_list.append(fit_task)
    
    fit_task = dask.delayed(utils.combine_into_dict)(task_list)
    # fit_task.visualize(filename = parallel_compute_graph_filename)
    
    main_info_logger.info('staring to fit ensemble '+ model_type +' models for ' + n_keys + ' groups')        
    #client = Client()  # start distributed scheduler locally.  Launch dashboard
    with ProgressBar():
        fit_results = fit_task.compute()
    
    t_ensemble_training_fit_end = time.time()
    main_info_logger.info('parallel ensemble '+ model_type +' fit runtime = ' + str((t_ensemble_training_fit_end - t_ensemble_training_fit_start)/60) + ' minutes')
    
    return fit_results


def parallel_pick_and_pass_winning_and_best_model(
                                                        fit_results_dict,
                                                        parallel_partitions,
                                                        model_list,
                                                        ensemble,
                                                        post_process_execution_path,
                                                        main_info_logger,
                                                        n_keys,
                                                        epsilon,
                                                        pre_processed_dict
                                                ):
    """

    :param fit_results_dict:
    :param parallel_partitions:
    :param model_list:
    :param ensemble:
    :param post_process_execution_path:
    :param main_info_logger:
    :param n_keys:
    :param epsilon:
    :param pre_processed_dict:
    :return:
    """
    t_model_selection_start = time.time()
    main_info_logger.info('staring parallel model selection ' + n_keys + ' groups')
    
    fit_results_dict_list = utils.chunk_dict(fit_results_dict, parallel_partitions)
    
    task_list = []
    for i in range(len(fit_results_dict_list)):
        fit_task = dask.delayed(model_selection.pick_winning_model)(
                                                                        fit_results_dict_list[i],
                                                                        model_list + ['ensemble'] if ensemble is True else model_list,
                                                                        epsilon
                                                                     )   
        task_list.append(fit_task)    
        fit_task = dask.delayed(utils.combine_into_dict)(task_list)
    
    with ProgressBar():
            fit_results_training = fit_task.compute()
    
    pre_processed_dict = model_selection.pass_winning_and_best_model(
                                                                             pre_processed_dict, 
                                                                             fit_results_training
                                                                        )
    
    t_model_selection_end = time.time()
    main_info_logger.info('parallel model selection runtime = ' + str((t_model_selection_end - t_model_selection_start)/60) + ' minutes')
    
    return (pre_processed_dict, fit_results_training)
