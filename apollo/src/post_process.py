"""
after model fit post processing functions
"""

import dask

import math
import numpy
from apollo.src import utils
from apollo.src.parallel_utils import get_partitioned_list, combine_into_df, parallel_apply_fun
from apollo.src import performance


def compile_results(
                        fit_results,
                        filled_in_and_extended_fact,
                        post_process_error_logger,
                        runtime_stats_info_logger,
                        root,
                        model_list,
                        pivot_event_calendar,
                        response_dict,
                        training_response,
                        holdout_response,
                        forecast_response,
                        key,
                        run_config,
                        time_grain,
                        run_id,
                        epsilon,
                        post_process_execution_path,
                        parallel_partition_key,
                        parallel_partitions,
                        time_series_attribs
                    ):
        """

        :param fit_results:
        :param filled_in_and_extended_fact:
        :param post_process_error_logger:
        :param runtime_stats_info_logger:
        :param root:
        :param model_list:
        :param pivot_event_calendar:
        :param response_dict:
        :param training_response:
        :param holdout_response:
        :param forecast_response:
        :param key:
        :param run_config:
        :param time_grain:
        :param run_id:
        :param epsilon:
        :param post_process_execution_path:
        :param parallel_partition_key:
        :param parallel_partitions:
        :param time_series_attribs:
        :return:
        """
        src.performance._runtime_stats(
                                        fit_results,
                                        model_list,
                                        runtime_stats_info_logger,
                                        root
                                      )        
        
        
        filled_in_and_extended_fact_columns = list(filled_in_and_extended_fact)
        filled_in_and_extended_fact_columns.extend(
                                                    [
                                                     'winning_model_forecast',
                                                     'best_model_forecast', 
                                                     'winning_model_wfa', 
                                                     'best_model_wfa', 
                                                     'benchmark_wfa',
                                                     'model_better',
                                                     'best_model',
                                                     'winning_model',
                                                     'abs_error_winning_model',
                                                     'abs_error_best_model',
                                                     'abs_error_benchmark'
                                                    ]
                                                  )
        filled_in_and_extended_fact = filled_in_and_extended_fact.reindex(columns = filled_in_and_extended_fact_columns)
        filled_in_and_extended_fact = filled_in_and_extended_fact.set_index(key)
        
        if post_process_execution_path == 'non_parallel':
            
            filled_in_and_extended_fact = _move_results_from_dict_to_df(
                                                                          filled_in_and_extended_fact = filled_in_and_extended_fact,
                                                                          fit_results = fit_results,
                                                                          post_process_error_logger = post_process_error_logger
                                                                      )
        
        if post_process_execution_path == 'parallel':
            
            filled_in_and_extended_fact = _parallel_move_results_from_dict_to_df(
                                                                                      filled_in_and_extended_fact = filled_in_and_extended_fact,
                                                                                      fit_results_dict = fit_results,
                                                                                      parallel_partitions = parallel_partitions,
                                                                                      post_process_error_logger = post_process_error_logger
                                                                                )
        
        filled_in_and_extended_fact = filled_in_and_extended_fact.reset_index().fillna(0)
        
        filled_in_and_extended_fact = _add_abs_error(
                                                        filled_in_and_extended_fact,
                                                        training_response,
                                                        holdout_response,
                                                        forecast_response
                                                    )
        
        filled_in_and_extended_fact = _drop_unused_and_overlapping_fields(
                                                                            filled_in_and_extended_fact,
                                                                            pivot_event_calendar,
                                                                            time_grain,
                                                                            post_process_execution_path,
                                                                            time_series_attribs,
                                                                            key,
                                                                            response_dict
                                                                         )
        
        filled_in_and_extended_fact.rename(columns = {training_response: 'training_response'}, inplace = True)
        filled_in_and_extended_fact['training_response_type'] = training_response
        if training_response != holdout_response:
            filled_in_and_extended_fact.rename(columns = {holdout_response: 'holdout_response'}, inplace = True)
            filled_in_and_extended_fact['holdout_response_type'] = holdout_response
        else:
            filled_in_and_extended_fact['holdout_response'] = filled_in_and_extended_fact['training_response']
            filled_in_and_extended_fact['holdout_response_type'] = filled_in_and_extended_fact['training_response_type']
                
        if post_process_execution_path == 'non_parallel':
            
            filled_in_and_extended_fact = _apply_bucket_wfa(filled_in_and_extended_fact.groupby(key), 'winning_model_wfa', 'holdout_winning_wfa_bucket')
            filled_in_and_extended_fact = _apply_bucket_wfa(filled_in_and_extended_fact.groupby(key), 'benchmark_wfa', 'holdout_benchmark_wfa_bucket')
            filled_in_and_extended_fact = _apply_bucket_wfa(filled_in_and_extended_fact.groupby(key), 'best_model_wfa', 'holdout_best_model_wfa_bucket')
            filled_in_and_extended_fact = filled_in_and_extended_fact.fillna(0)
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(_add_apollo_class)
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(_add_apollo_wfa_gap)
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(_add_apollo_wfa_gap_buckets)
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(
                                                                                            _bucket_benchmark_future_forecast_deviation,
                                                                                            epsilon,
                                                                                            'best_model_'
                                                                                        )        
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(
                                                                                            _bucket_benchmark_future_forecast_deviation,
                                                                                            epsilon,
                                                                                            'winning_model_'
                                                                                        )
        
        if  post_process_execution_path == 'parallel':
            
            dask.config.set(scheduler='processes')
            
            filled_in_and_extended_fact = _parallel_apply_bucket_wfa(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), 'winning_model_wfa', 'holdout_winning_wfa_bucket', key)
            filled_in_and_extended_fact = _parallel_apply_bucket_wfa(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), 'benchmark_wfa', 'holdout_benchmark_wfa_bucket', key)
            filled_in_and_extended_fact = _parallel_apply_bucket_wfa(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), 'best_model_wfa', 'holdout_best_model_wfa_bucket', key)
            filled_in_and_extended_fact = filled_in_and_extended_fact.fillna(0)
            filled_in_and_extended_fact = parallel_apply_fun(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), _add_apollo_class, key)
            filled_in_and_extended_fact = parallel_apply_fun(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), _add_apollo_wfa_gap, key)
            filled_in_and_extended_fact = parallel_apply_fun(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), _add_apollo_wfa_gap_buckets, key)
            filled_in_and_extended_fact = _parallel_apply_bucket_benchmark_future_forecast_deviation(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), epsilon, 'best_model_', key)
            filled_in_and_extended_fact = _parallel_apply_bucket_benchmark_future_forecast_deviation(get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key), epsilon, 'winning_model_', key)
        
        if post_process_execution_path == 'parallel':
            filled_in_and_extended_fact = filled_in_and_extended_fact.drop('parallel_partition_dense_rank', axis = 1)
        
        filled_in_and_extended_fact['run_id'] = run_id
                
        return filled_in_and_extended_fact   



def _add_abs_error(
                    filled_in_and_extended_fact,
                    training_response,
                    holdout_response,
                    forecast_response
                  ):
    """

    :param filled_in_and_extended_fact:
    :param training_response:
    :param holdout_response:
    :param forecast_response:
    :return:
    """
    filled_in_and_extended_fact['response_for_abs_error'] = filled_in_and_extended_fact[training_response]
    filled_in_and_extended_fact[filled_in_and_extended_fact['data_split'] == 'Holdout'][['response_for_abs_error']] = filled_in_and_extended_fact[filled_in_and_extended_fact['data_split'] == 'Holdout'][[holdout_response]]
    filled_in_and_extended_fact[filled_in_and_extended_fact['data_split'] == 'Forecast'][['response_for_abs_error']] = filled_in_and_extended_fact[filled_in_and_extended_fact['data_split'] == 'Forecast'][[forecast_response]]
    
    for i in (['winning_model', 'best_model', 'benchmark']):
        if i == 'benchmark':
            suffix = ''
        else:
            suffix = '_forecast'
        filled_in_and_extended_fact[str(i) + suffix] = filled_in_and_extended_fact[str(i) + suffix].clip(lower = 0)
        filled_in_and_extended_fact['abs_error_' + str(i)] = (filled_in_and_extended_fact[str(i) + suffix] - filled_in_and_extended_fact['response_for_abs_error']).abs()
        filled_in_and_extended_fact['abs_error_' + str(i)] = filled_in_and_extended_fact['abs_error_' + str(i)].fillna(0).astype(int)
    
    return filled_in_and_extended_fact



def _drop_unused_and_overlapping_fields(
                                            filled_in_and_extended_fact,
                                            pivot_event_calendar,
                                            time_grain,
                                            post_process_execution_path,
                                            time_series_attribs,
                                            key,
                                            response_dict
                                       ):
    """

    :param filled_in_and_extended_fact:
    :param pivot_event_calendar:
    :param time_grain:
    :param post_process_execution_path:
    :param time_series_attribs:
    :param key:
    :param response_dict:
    :return:
    """
    result_drop_fields = list(pivot_event_calendar)
    if time_grain == 'week':
        result_drop_fields.remove('division')
    result_drop_fields.remove(time_grain)
    if post_process_execution_path == 'non_parallel':
        result_drop_fields.append('parallel_partition_dense_rank')
    result_drop_fields.append('forecast_length')
    result_drop_fields.append('holdout_length')
    result_drop_fields.append('holdout_shifted_response')
    result_drop_fields.append('forecast_shifted_response')
    result_drop_fields.append('response_for_abs_error')
    time_series_attribs_fields = list(time_series_attribs)
    time_series_attribs_fields = [n for n in time_series_attribs_fields if n not in key + ['index'] + ['run_id']]
    result_drop_fields = result_drop_fields + time_series_attribs_fields
    
    shifted_fields = []
    for k, value in response_dict.items():
        shifted_fields = shifted_fields + ['shifted_' + value]
    result_drop_fields = result_drop_fields + shifted_fields
    
    filled_in_and_extended_fact.drop(
                                        labels = result_drop_fields, 
                                        axis = 1,
                                        inplace = True,
                                        errors = 'ignore'
                                    )

    return filled_in_and_extended_fact

def _move_results_from_dict_to_df(filled_in_and_extended_fact, fit_results, post_process_error_logger):
    """

    :param filled_in_and_extended_fact:
    :param fit_results:
    :param post_process_error_logger:
    :return:
    """
    for fit_results_key, fit_results_value in fit_results.items():
        
        for m in ['winning_model', 'best_model', 'benchmark']:
              try:
                  
                  if m != 'benchmark':
                  
                      if fit_results_value[m] is not None and fit_results_value[m] is not None:
        
                          model_estimate = fit_results_value[m + '_fitted_training'].append(fit_results_value[m + '_holdout']).append(fit_results_value[m + '_forecast'])
                          filled_in_and_extended_fact.loc[[fit_results_key], [m + '_forecast']] = model_estimate.values
                          filled_in_and_extended_fact[m + '_forecast'] = filled_in_and_extended_fact[m + '_forecast'].fillna(0).astype(int)                      
                          model_wfa = fit_results_value[m + '_wfa']
                          filled_in_and_extended_fact.loc[[fit_results_key], [m + '_wfa']] = model_wfa
                          filled_in_and_extended_fact.loc[[fit_results_key], [m]] = fit_results_value[m]
                  
                  else:
                      
                          model_wfa = fit_results_value[m + '_wfa']
                          filled_in_and_extended_fact.loc[[fit_results_key], [m + '_wfa']] = model_wfa
    
              except Exception as e:
                  post_process_error_logger.error('error post processing ' + m + ' for ' + ' '.join(fit_results_key) + ' with error ' + str(e))              
              
              if m == 'winning_model':
                  filled_in_and_extended_fact.loc[[fit_results_key], ['model_better']] = fit_results_value['model_better']
                
    return filled_in_and_extended_fact



def combine_fit_result_dicts(
                                fit_results_training,
                                fit_results_complete
                            ):
    """

    :param fit_results_training:
    :param fit_results_complete:
    :return:
    """
    fit_results = {}        
    for d in [fit_results_training, fit_results_complete]:            
        for k, v in d.items():                
            fit_results.setdefault(k, {}).update(v)
    
    return fit_results



def _bucket_wfa(
                x,
                wfa_field,
                wfa_bucket_field
               ):
    """

    :param x:
    :param wfa_field:
    :param wfa_bucket_field:
    :return:
    """
    holdout_data = x[x['data_split'] == 'Holdout']
    if holdout_data.empty is False:
        
        holdout_wfa = numpy.nan_to_num(holdout_data[wfa_field].sample(1).values.item(0))
        lower = min(80, math.floor(holdout_wfa*100) - math.floor(holdout_wfa*100) % 20)
        upper = min(100, lower + 20)
        wfa_bucket = str(lower) + ' - ' + str(upper) + '%'
        x[wfa_bucket_field] = wfa_bucket
        
    else:
        x[wfa_bucket_field] = 'No Holdout'
    
    return x


def _apply_bucket_wfa(
                         gb,
                         wfa_field,
                         wfa_bucket_field                     
                     ):
    """

    :param gb:
    :param wfa_field:
    :param wfa_bucket_field:
    :return:
    """
    x = gb.apply(_bucket_wfa, wfa_field, wfa_bucket_field)
    
    return x




def _add_apollo_class(x):
    """

    :param x:
    :return:
    """
    model_better = x['model_better'].sample(1).values.item(0)
    holdout_winning_wfa_bucket = x['holdout_winning_wfa_bucket'].sample(1).values.item(0)
    time_series_class = x['time_series_class'].sample(1).values.item(0)
    
    if model_better == 'Y' and (holdout_winning_wfa_bucket == '60 - 80%' or holdout_winning_wfa_bucket == '80 - 100%'):
        x['apollo_class'] = 'High Performance'
    elif model_better == 'N' and holdout_winning_wfa_bucket != '0 - 20%' and time_series_class != 'not_enough_history':
        x['apollo_class'] = 'Apollo Opportunity'
    elif model_better == 'Y' and holdout_winning_wfa_bucket != '60 - 80%' and holdout_winning_wfa_bucket != '80 - 100%':
        x['apollo_class'] = 'Investigate'
    elif model_better == 'N' and holdout_winning_wfa_bucket == '0 - 20%':
        x['apollo_class'] = 'Dogs'
    
    return x


def _add_apollo_wfa_gap(x):
    """

    :param x:
    :return:
    """
    x['apollo_wfa_gap'] = x['best_model_wfa'] - x['benchmark_wfa']
    
    return x


def _add_apollo_wfa_gap_buckets(x):
    """

    :param x:
    :return:
    """
    y = x[x['data_split'] == 'Holdout']['apollo_wfa_gap']
    
    if y.shape[0] > 0:
        z = y.sample(1).values.item(0)
        if z <= -0.5:
            x['apollo_wfa_gap_bucket'] = '(-6) <= -50%'
        elif -0.5 < z <= -0.25:
            x['apollo_wfa_gap_bucket'] = '(-5) -50% to -25%'
        elif -0.25 < z <= -0.15:
            x['apollo_wfa_gap_bucket'] = '(-4) -25% to -15%'
        elif -0.15 < z <= -0.10:
            x['apollo_wfa_gap_bucket'] = '(-3) -15% to -10%'
        elif -0.10 < z <= -0.05:
            x['apollo_wfa_gap_bucket'] = '(-2) -10% to -5%'
        elif -0.05 < z < 0:
            x['apollo_wfa_gap_bucket'] = '(-1) -5% to 0%'
        elif 0 <= z <= 0.05:
            x['apollo_wfa_gap_bucket'] = '(1) 0% to 5%'
        elif 0.05 < z <= 0.10:
            x['apollo_wfa_gap_bucket'] = '(2) 5% to 10%'
        elif 0.10 < z <= 0.15:
            x['apollo_wfa_gap_bucket'] = '(3) 10% to 15%'
        elif 0.15 < z <= 0.25:
            x['apollo_wfa_gap_bucket'] = '(4) 15% to 25%'
        elif 0.25 < z <= 0.50:
            x['apollo_wfa_gap_bucket'] = '(5) 25% to 50%'
        else:
            x['apollo_wfa_gap_bucket'] = '(6) > 50%'
    else:
        x['apollo_wfa_gap_bucket'] = ''
    
    return x


def _bucket_benchmark_future_forecast_deviation(
                                                    x,
                                                    epsilon,
                                                    forecast_field
                                               ):
    """

    :param x:
    :param epsilon:
    :param forecast_field:
    :return:
    """
    forecast_data = x[x['data_split'] == 'Forecast']
    if forecast_data.empty is False:
        apollo_benchmark_future_forecast_deviation = abs(sum(abs(forecast_data[forecast_field + 'forecast'] - forecast_data['benchmark']))/(epsilon + sum(forecast_data['benchmark'])))
        if apollo_benchmark_future_forecast_deviation >= 1:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(1) > 100%'
        elif 0.5 <= apollo_benchmark_future_forecast_deviation < 1:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(2) 50% <= x <= 100%'
        elif 0.25 <= apollo_benchmark_future_forecast_deviation < 0.5:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(3) 25% <= x < 50%'
        elif 0.15 <= apollo_benchmark_future_forecast_deviation < 0.25:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(4) 15% <= x < 25%'
        elif 0.10 <= apollo_benchmark_future_forecast_deviation < 0.15:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(5) 10% <= x < 15%'
        elif 0.05 <= apollo_benchmark_future_forecast_deviation < 0.10:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(6) 5% <= x < 10%'
        elif 0 <= apollo_benchmark_future_forecast_deviation < 0.05:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(7) 0% <= x < 5%'
        else:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(8) other'
    else:
            x[forecast_field + 'benchmark_future_forecast_deviation_bucket'] = '(8) other'
    
    return x


def _parallel_apply_bucket_wfa(
                                pre_processed_data_set_list,
                                wfa_field,
                                wfa_bucket_field,
                                key
                              ):
    """

    :param pre_processed_data_set_list:
    :param wfa_field:
    :param wfa_bucket_field:
    :param key:
    :return:
    """
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        gb = pre_processed_data_set_list[i][1].groupby(key)
        fit_task = dask.delayed(_apply_bucket_wfa)(gb, wfa_field, wfa_bucket_field)
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    filled_in_and_extended_fact  = fit_task.compute()
    
    return filled_in_and_extended_fact

def _parallel_apply_bucket_benchmark_future_forecast_deviation(
                                                                pre_processed_data_set_list,
                                                                epsilon,
                                                                forecast_field,
                                                                key
                                                              ):
    """

    :param pre_processed_data_set_list:
    :param epsilon:
    :param forecast_field:
    :param key:
    :return:
    """
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        gb = pre_processed_data_set_list[i][1].groupby(key)
        fit_task = dask.delayed(_apply_bucket_benchmark_future_forecast_deviation)(gb, epsilon, forecast_field)
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    filled_in_and_extended_fact  = fit_task.compute()
    
    return filled_in_and_extended_fact


def _apply_bucket_benchmark_future_forecast_deviation(gb, epsilon, forecast_field):
    """

    :param gb:
    :param epsilon:
    :param forecast_field:
    :return:
    """
    x = gb.apply(_bucket_benchmark_future_forecast_deviation, epsilon, forecast_field)
    
    return x


def _parallel_move_results_from_dict_to_df(
                                                filled_in_and_extended_fact,
                                                fit_results_dict,
                                                parallel_partitions,
                                                post_process_error_logger
                                           ):
    """

    :param filled_in_and_extended_fact:
    :param fit_results_dict:
    :param parallel_partitions:
    :param post_process_error_logger:
    :return:
    """
    task_list = []
    fit_results_dict_list = src.utils.chunk_dict(fit_results_dict, parallel_partitions)
    for i in range(len(fit_results_dict_list)):      
        df = filled_in_and_extended_fact.loc[list(fit_results_dict_list[i].keys())]
        fit_task = dask.delayed(_move_results_from_dict_to_df)(
                                                                df, 
                                                                fit_results_dict_list[i], 
                                                                post_process_error_logger
                                                              )
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    filled_in_and_extended_fact  = fit_task.compute()
    
    return filled_in_and_extended_fact
