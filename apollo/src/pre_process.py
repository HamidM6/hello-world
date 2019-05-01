"""
pre process and prep data for model fit
"""

from apollo.src import cleanse_history
import dask
import datetime
import numpy
import os
import pandas
from apollo.src.parallel_utils import get_partitioned_list, combine_into_df
from apollo.src import utils
from apollo.src import memory_mgmt

def read_in_data(
                    run_config,
                    main_info_logger,
                    time_grain,
                    key,
                    data_filters,
                    response_dict,
                    training_response,
                    holdout_response,
                    forecast_response,
                    pass_through_key_attributes,
                    benchmark
                ):
    """

    :param run_config:
    :param main_info_logger:
    :param time_grain:
    :param key:
    :param data_filters:
    :param response_dict:
    :param training_response:
    :param holdout_response:
    :param forecast_response:
    :param pass_through_key_attributes:
    :param benchmark:
    :return:
    """
    data_store = run_config['DATA_STORE']   
    parallel_partitions = run_config['PARALLEL_PARTITIONS']
    
    main_info_logger.info('reading data store queries...')
    query_dictionary = src.utils.read_params_in_from_json('queries.json')
    root = os.path.dirname(os.getcwd())
    
    field_list = [x.strip().lower() for x in list(data_filters.keys())]
    filter_list = ["" if x == [] else [y.strip().lower() for y in x] for x in list(data_filters.values())]
    
    for query in query_dictionary:       
        
        query_location = root + '\\sql' + query_dictionary[query]
        sql = src.utils.read_in_file(query_location)
        
        if query == 'FACT':
            
            main_info_logger.info('reading fact table...')
            
            raw_fact = src.connect.get_data(
                                            data_store = data_store,
                                            query = sql
                                            )
            main_info_logger.info('read in '
                                     + str(raw_fact.groupby(key).ngroups)
                                     + ' '
                                     + ' '.join(key)
                                     + ' time series '
                                     + ' for a total of '
                                     + str(len(raw_fact.index))
                                     + ' rows'
                                     )
            raw_fact = lower_case_and_filter(
                                                data = raw_fact,
                                                field_list = field_list,
                                                filter_list = filter_list
                                            )
            raw_fact[time_grain] = pandas.to_datetime(raw_fact[time_grain])
            raw_fact = drop_unused_response(
                                                data = raw_fact,
                                                response_dict = response_dict
                                            )
            raw_fact = drop_unused_benchmark(
                                                data = raw_fact,
                                                benchmark = benchmark
                                            )
            raw_fact = aggregate_by_key(
                                            data = raw_fact,
                                            key = key,
                                            time_grain = time_grain,
                                            response_dict = response_dict,
                                            benchmark = benchmark,
                                            pass_through_key_attributes = pass_through_key_attributes,
                                            parallel_partitions = parallel_partitions
                                        )
            raw_fact = src.memory_mgmt.optimize_raw_fact_memory(
                                                                raw_fact = raw_fact,
                                                                response_dict = response_dict,
                                                                key = key
                                                           )
            main_info_logger.info('filtered down to '
                                     + str(raw_fact.groupby(key).ngroups)
                                     + ' '
                                     + ' '.join(key)
                                     + ' time series '
                                     + ' for a total of '
                                     + str(len(raw_fact.index))
                                     + ' rows'
                                     )
            
        elif query == "EVENT_CALENDAR":
            
            main_info_logger.info('reading event calendar...')
            
            event_calendar = src.connect.get_data(
                                                    data_store = data_store,
                                                    query = sql
                                                    )      
            if time_grain == 'week':
                
                if len(data_filters['division']) == 0:
                    f = [""]
                else:
                    f = [[x.strip().lower() for x in data_filters['division']]]
                
                event_calendar = lower_case_and_filter(
                                                        data = event_calendar,
                                                        field_list = ['division'],
                                                        filter_list = f
                                                      )
            event_calendar[time_grain] = pandas.to_datetime(event_calendar[time_grain])
            
            
    main_info_logger.info('data read complete...')
    
    return raw_fact, event_calendar




def find_max_date(data, time_grain, training_response):
    """

    :param data:
    :param time_grain:
    :param training_response:
    :return:
    """
    agg_by_time_grain = data[data[time_grain] < datetime.datetime.today().strftime('%Y-%m-%d')].groupby(time_grain).agg('sum')[training_response]
    agg_by_time_grain = agg_by_time_grain.reset_index()
    max_date = datetime.datetime.strptime(str(agg_by_time_grain[agg_by_time_grain[training_response] > 0][time_grain].agg('max').date()), '%Y-%m-%d')
    return max_date



def prepare_for_model_fit(
                             raw_fact,
                             model_param_config,
                             main_info_logger,
                             event_calendar,
                             benchmark,
                             key,
                             time_grain,
                             training_response,
                             forecast_response,
                             response_dict,
                             run_id,
                             pass_through_key_attributes,
                             pre_process_execution_path,
                             parallel_partition_key,
                             epsilon,
                             cap_outliers
                         ):
    """

    :param raw_fact:
    :param model_param_config:
    :param main_info_logger:
    :param event_calendar:
    :param benchmark:
    :param key:
    :param time_grain:
    :param training_response:
    :param forecast_response:
    :param response_dict:
    :param run_id:
    :param pass_through_key_attributes:
    :param pre_process_execution_path:
    :param parallel_partition_key:
    :param epsilon:
    :param cap_outliers:
    :return:
    """
    stdevs_to_cap_outliers = model_param_config['STDEVS_TO_CAP_OUTLIERS']
    max_date = model_param_config['MAX_DATE']      
    
    if max_date == "":
        max_date = find_max_date(
                                    data = raw_fact,
                                    time_grain = time_grain,
                                    training_response = training_response
                                )
    else:
        max_date = datetime.datetime.strptime(model_param_config['MAX_DATE'], '%Y-%m-%d')
    zero_weeks_to_disco = model_param_config['ZERO_WEEKS_TO_DISCO']
    min_history_to_forecast = model_param_config['MIN_HISTORY_TO_FORECAST']
    holdout_start_date = datetime.datetime.strptime(model_param_config['HOLDOUT_START_DATE'], '%Y-%m-%d')
    holdout_to_training_ratio_new_item_threshold = model_param_config['HOLDOUT_TO_TRAINING_RATIO_NEW_ITEM_THRESHOLD']
    holdout_to_training_ratio_disco_threshold = model_param_config['HOLDOUT_TO_TRAINING_RATIO_DISCO_THRESHOLD']

    if time_grain == 'week':
        event_calendar_key = [time_grain, 'division']
    elif time_grain == 'month':
        event_calendar_key = [time_grain]
    
    main_info_logger.info('pre_process: read in config params...')
    
    if pre_process_execution_path == 'non_parallel':    
    
        time_series_attribs = src.time_series_attribs.get_time_series_attribs(
                                                                              time_series_data_frame = raw_fact, 
                                                                              key = key, 
                                                                              time_grain = time_grain, 
                                                                              training_response = training_response,
                                                                              max_date = max_date,
                                                                              zero_weeks_to_disco = zero_weeks_to_disco,
                                                                              epsilon = epsilon,
                                                                              holdout_start_date = holdout_start_date
                                                                              )
    elif pre_process_execution_path == 'parallel':
        
        pre_processed_data_set_list = get_partitioned_list(raw_fact, parallel_partition_key)      
        time_series_attribs = _parallel_get_time_series_attribs(
                                                                    pre_processed_data_set_list, 
                                                                    key, 
                                                                    time_grain, 
                                                                    training_response,
                                                                    max_date,
                                                                    zero_weeks_to_disco,
                                                                    epsilon,
                                                                    holdout_start_date
                                                               )
    
    main_info_logger.info('pre_process: computed and merged time series attributes...')
    
    if pre_process_execution_path == 'non_parallel': 
    
        filled_in_and_extended_fact = src.cleanse_history.fill_in_missing_periods(
                                                                                    time_series_data_frame = raw_fact, 
                                                                                    key = key,
                                                                                    time_grain = time_grain,
                                                                                    response_dict = response_dict,
                                                                                    pass_through_key_attributes = pass_through_key_attributes
                                                                                )
    
    elif pre_process_execution_path == 'parallel': 
        
        filled_in_and_extended_fact = _parallel_fill_in_missing_periods(
                                                                           pre_processed_data_set_list = pre_processed_data_set_list, 
                                                                           key = key,
                                                                           time_grain = time_grain,
                                                                           response_dict = response_dict,
                                                                           pass_through_key_attributes = pass_through_key_attributes
                                                                        )
    
    main_info_logger.info('pre_process: filled in missing periods...')
    
    filled_in_and_extended_fact = filled_in_and_extended_fact.reset_index()
        
    time_series_attribs = time_series_attribs.reset_index()
    filled_in_and_extended_fact = filled_in_and_extended_fact.merge(
                                                                    time_series_attribs,
                                                                    how = 'left',
                                                                    on = key
                                                                    )
    
    filled_in_and_extended_fact['time_series_class'] = 'nominal'    

    filled_in_and_extended_fact['data_split'] = filled_in_and_extended_fact[time_grain].apply(
                                                                                                src.time_series_attribs.time_series_sample_labeling_conditions,
                                                                                                args = (max_date, holdout_start_date)
                                                                                              )

    if cap_outliers:
        filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(
                                                                                        src.cleanse_history.machine_cleanse_response,
                                                                                        training_response,
                                                                                        forecast_response,
                                                                                        stdevs_to_cap_outliers
                                                                                    )

    
    
    main_info_logger.info('pre_process: computed capped response...')

    filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(
                                                                                    _shift_response,
                                                                                    time_grain,
                                                                                    response_dict
                                                                                    )
    
    
    filled_in_and_extended_fact = _add_benchmark(
                                                   filled_in_and_extended_fact = filled_in_and_extended_fact,  
                                                   benchmark = benchmark,
                                                   response_dict = response_dict
                                                )
    
#################### filter discos ##################
    
    total_groups = filled_in_and_extended_fact.groupby(key).ngroups
    
    if total_groups > 0:
    
        discos = filled_in_and_extended_fact[filled_in_and_extended_fact['is_disco'] == True].groupby(key)['is_disco'].max().reset_index()
        time_series_attribs = time_series_attribs.merge(
                                                         discos,
                                                         how = 'left',
                                                         on = key,
                                                         suffixes = ('_y', '')
                                                        ).fillna(False)
        time_series_attribs.drop('is_disco_y', axis = 1, inplace = True)
        filled_in_and_extended_fact['time_series_class'].loc[filled_in_and_extended_fact['is_disco'] == True] = 'is_disco'
        groups_without_discos = filled_in_and_extended_fact[filled_in_and_extended_fact['is_disco'] == False].groupby(key).ngroups
        main_info_logger.info('identified and labeled ' + str(groups_without_discos) + ' non-disco groups')    
    
#################### filter groups without enough history ##################
    
        filled_in_and_extended_fact['time_series_class'].loc[(filled_in_and_extended_fact['time_series_length'] <= min_history_to_forecast) & (filled_in_and_extended_fact['time_series_class'] == 'nominal')] = 'not_enough_history'
        groups_without_enough_history = filled_in_and_extended_fact[filled_in_and_extended_fact['time_series_class'] == 'not_enough_history'].groupby(key).ngroups
        main_info_logger.info('identified and labeled ' + str(groups_without_enough_history) + ' additional groups without enough history')
        time_series_attribs['not_enough_history'] = time_series_attribs['time_series_length'].apply(lambda x: True if x <= min_history_to_forecast else False)
    
#################### filter groups without enough history and near disco items ##################
            
        pivot_filled_in_and_extended_fact = (
                                                pandas.pivot_table(
                                                                    data = filled_in_and_extended_fact,
                                                                    columns = 'data_split',
                                                                    aggfunc = numpy.mean,
                                                                    index = key,
                                                                    values = training_response,
                                                                    fill_value = 0
                                                                 ).reset_index()
                                            ).drop('Forecast', axis = 1, errors = 'ignore')
        pivot_filled_in_and_extended_fact['holdout_training_ratio'] = pivot_filled_in_and_extended_fact['Holdout'].divide(pivot_filled_in_and_extended_fact['Training'] + epsilon)
        pivot_filled_in_and_extended_fact = pivot_filled_in_and_extended_fact.drop(['Holdout', 'Training'], axis = 1)
        filled_in_and_extended_fact = filled_in_and_extended_fact.merge(
                                                                        pivot_filled_in_and_extended_fact,
                                                                        how = 'left',
                                                                        on = key
                                                                        )
        filled_in_and_extended_fact['time_series_class'].loc[(filled_in_and_extended_fact['holdout_training_ratio'] >= holdout_to_training_ratio_new_item_threshold) & (filled_in_and_extended_fact['time_series_class'] == 'nominal')] = 'new_item'
        groups_with_new_items = filled_in_and_extended_fact[filled_in_and_extended_fact['time_series_class'] == 'new_item'].groupby(key).ngroups
        main_info_logger.info('identified and labeled ' + str(groups_with_new_items) + ' new item groups')
 
        filled_in_and_extended_fact['time_series_class'].loc[(filled_in_and_extended_fact['holdout_training_ratio'] <= holdout_to_training_ratio_disco_threshold) & (filled_in_and_extended_fact['time_series_class'] == 'nominal')] = 'near_disco'
        groups_with_near_disco_items = filled_in_and_extended_fact[filled_in_and_extended_fact['time_series_class'] == 'near_disco'].groupby(key).ngroups
        main_info_logger.info('identified and labeled ' + str(groups_with_near_disco_items) + ' near disco groups')    
        time_series_attribs = time_series_attribs.merge(
                                                         pivot_filled_in_and_extended_fact,
                                                         how = 'left',
                                                         on = key
                                                        )
        time_series_attribs['new_item'] = time_series_attribs['holdout_training_ratio'].apply(lambda x: True if x >= holdout_to_training_ratio_new_item_threshold else False)
        time_series_attribs['near_disco'] = time_series_attribs['holdout_training_ratio'].apply(lambda x: True if x <= holdout_to_training_ratio_disco_threshold else False)

#################### filter intermittent ##################
        
        percent_zeros_threshold_for_intermittent = model_param_config['PERCENT_ZEROS_THRESHOLD_FOR_INTERMITTENT']
        
        if pre_process_execution_path == 'non_parallel':  
            filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(_check_intermittent, training_response, percent_zeros_threshold_for_intermittent)
        
        if pre_process_execution_path == 'parallel':          
            pre_processed_data_set_list = get_partitioned_list(filled_in_and_extended_fact, parallel_partition_key) 
            filled_in_and_extended_fact = _parallel_check_intermittent(
                                                                          pre_processed_data_set_list,
                                                                          key,
                                                                          training_response,
                                                                          percent_zeros_threshold_for_intermittent
                                                                      )
        
        main_info_logger.info('labelled intermittent time series...')
        
#################### join event calendar ##################
    
        pivot_event_calendar = src.cleanse_history.pivot_and_fill_in_missing_periods(event_calendar, time_grain, event_calendar_key)
        
        cols = pivot_event_calendar.columns[pivot_event_calendar.dtypes.eq('float64')]
        if len(cols) > 0:
            pivot_event_calendar[cols] = pivot_event_calendar[cols].astype('int8')
        
        
        filled_in_and_extended_fact = filled_in_and_extended_fact.merge(
                                                                            right = pivot_event_calendar,
                                                                            how = 'left',
                                                                            on = event_calendar_key,
                                                                            copy = False,
                                                                            indicator = False
                                                                        )
        filled_in_and_extended_fact = filled_in_and_extended_fact.fillna(0)   
		
#################### append holdout and forecast part of event calendar to filled_in_and_extended_fact ##################
		

        exog_cols = list(pivot_event_calendar)
        [exog_cols.remove(col) for col in event_calendar_key]
#
#        filled_in_and_extended_fact = filled_in_and_extended_fact.groupby(key).apply(_append_exogs_to_df, exog_cols)
#
#        exog_cols_holdout = [col + '_holdout' for col in exog_cols]
#        exog_cols_forecast = [col + '_forecast' for col in exog_cols]
#        
#        filled_in_and_extended_fact.columns = orig_cols + exog_cols + exog_cols_holdout + exog_cols_forecast
		
                
#################### clean up time_series_attribs for write ##################
        
        
        time_series_attribs['run_id'] = run_id    
        time_series_attribs = time_series_attribs.reset_index()    
        time_series_attribs = time_series_attribs.fillna(0)
        time_series_attribs['coeff_of_variation'] = numpy.where(time_series_attribs['coeff_of_variation'] == False, 0, time_series_attribs['coeff_of_variation'])
        time_series_attribs['long_term_stdev'] = numpy.where(time_series_attribs['long_term_stdev'] == False, 0, time_series_attribs['long_term_stdev'])    
        
        for k, response in response_dict.items():
            filled_in_and_extended_fact[response] = filled_in_and_extended_fact[response].clip(lower = 0)
        
    else:
        filled_in_and_extended_fact[response] = pandas.DataFrame()

    if time_grain == 'week':
        div = filled_in_and_extended_fact['division'].sample(1).values.item(0)                
        holidays = event_calendar[event_calendar.division == div][[time_grain, 'event_name']].reset_index(drop=True)
        holidays.columns = ['ds', 'holiday']
    else:
        holidays = event_calendar[[time_grain, 'event_name']]
        holidays.columns = ['ds', 'holiday']
    
    main_info_logger.info('data cleanse complete...')
    
    return filled_in_and_extended_fact, pivot_event_calendar, time_series_attribs, holidays, exog_cols


def lower_case_and_filter(data, field_list, filter_list):
    """

    :param data:
    :param field_list:
    :param filter_list:
    :return:
    """
    # add additional actions to action_list and associated arguments    
    
    for i, action in enumerate(field_list):
                
        if field_list[i] in data.columns:
            data[field_list[i]] = data[field_list[i]].str.lower()
            data[field_list[i]] = data[field_list[i]].str.strip()
        if field_list[i] in data.columns:
            data = data if filter_list[i] == "" else data[data[field_list[i]].isin(filter_list[i])]        
        
    return data



def drop_unused_response(data, response_dict):
    """

    :param data:
    :param response_dict:
    :return:
    """
    for k, response in response_dict.items():
        
        if 'pos_sales' in response:
            data.drop('clean_pos_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_pos_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('pos_unit', inplace = True, axis = 1, errors = 'ignore')
        if 'pos_unit' in response:
            data.drop('clean_pos_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_pos_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('pos_sales', inplace = True, axis = 1, errors = 'ignore')
        if 'order_sales' in response:
            data.drop('clean_order_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_order_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('order_unit', inplace = True, axis = 1, errors = 'ignore')
        if 'order_unit' in response:
            data.drop('clean_order_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_order_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('order_sales', inplace = True, axis = 1, errors = 'ignore')
        if 'invoice_sales' in response:
            data.drop('clean_invoice_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_invoice_unit', inplace = True, axis = 1, errors = 'ignore')
            data.drop('invoice_unit', inplace = True, axis = 1, errors = 'ignore')
        if 'invoice_unit' in response:
            data.drop('clean_invoice_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('unclean_invoice_sales', inplace = True, axis = 1, errors = 'ignore')
            data.drop('invoice_sales', inplace = True, axis = 1, errors = 'ignore')
    
    return data



def drop_unused_benchmark(data, benchmark):
    """

    :param data:
    :param benchmark:
    :return:
    """
    if benchmark == 'stat':
        data.drop('consensus', inplace = True, axis = 1, errors = 'ignore')
    if benchmark == 'consensus':
        data.drop('stat', inplace = True, axis = 1, errors = 'ignore')
    
    return data



def aggregate_by_key(
                        data, 
                        key, 
                        time_grain, 
                        response_dict,
                        benchmark, 
                        pass_through_key_attributes,
                        parallel_partitions
                        ):
    """

    :param data:
    :param key:
    :param time_grain:
    :param response_dict:
    :param benchmark:
    :param pass_through_key_attributes:
    :param parallel_partitions:
    :return:
    """
    agg_list = []
    for k, response in response_dict.items():
        agg_list.append(response)
    gb_key = key + [time_grain] + pass_through_key_attributes
    if benchmark != 'last_year':
        agg_list.append(benchmark)        
    data = data.groupby(gb_key)[agg_list].agg('sum')
    data = data.reset_index()
    data['parallel_partition_dense_rank'] = data.groupby(key).ngroup()
    data['parallel_partition_dense_rank'] = data['parallel_partition_dense_rank'] % parallel_partitions
    
    return data
    
def _check_intermittent(
                            x, 
                            training_response,
                            percent_zeros_threshold_for_intermittent
                        ):
    """

    :param x:
    :param training_response:
    :param percent_zeros_threshold_for_intermittent:
    :return:
    """
    time_series_class = x['time_series_class'].sample(1).values.item(0)
    
    if time_series_class == 'nominal':
        
        complete_x = x[x['data_split'] != 'Forecast']
        n = complete_x.shape[0]
        
        if n > 0:
            zeros_ratio = complete_x[complete_x[training_response] == 0].shape[0] / n
            if zeros_ratio >= percent_zeros_threshold_for_intermittent:
                x['time_series_class'] = 'intermittent'    
    return x


def _parallel_get_time_series_attribs(
                                        pre_processed_data_set_list, 
                                        key, 
                                        time_grain, 
                                        training_response,
                                        max_date,
                                        zero_weeks_to_disco,
                                        epsilon,
                                        holdout_start_date
                                     ):
    """

    :param pre_processed_data_set_list:
    :param key:
    :param time_grain:
    :param training_response:
    :param max_date:
    :param zero_weeks_to_disco:
    :param epsilon:
    :param holdout_start_date:
    :return:
    """
    dask.config.set(scheduler='processes')
    
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        df_list_element = pre_processed_data_set_list[i]
        df = df_list_element[1]
        fit_task = dask.delayed(src.time_series_attribs.get_time_series_attribs)(
                                                                                    df, 
                                                                                    key, 
                                                                                    time_grain, 
                                                                                    training_response,
                                                                                    max_date,
                                                                                    zero_weeks_to_disco,
                                                                                    epsilon,
                                                                                    holdout_start_date
                                                                                )
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    time_series_attribs  = fit_task.compute()
    
    return time_series_attribs


def _parallel_fill_in_missing_periods(
                                        pre_processed_data_set_list, 
                                        key,
                                        time_grain,
                                        response_dict,
                                        pass_through_key_attributes
                                     ):
    """

    :param pre_processed_data_set_list:
    :param key:
    :param time_grain:
    :param response_dict:
    :param pass_through_key_attributes:
    :return:
    """
    dask.config.set(scheduler='processes')
    
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        df_list_element = pre_processed_data_set_list[i]
        df = df_list_element[1]
        fit_task = dask.delayed(src.cleanse_history.fill_in_missing_periods)(
                                                                                df, 
                                                                                key,
                                                                                time_grain,
                                                                                response_dict,
                                                                                pass_through_key_attributes
                                                                            )
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    filled_in_and_extended_fact  = fit_task.compute()
    
    return filled_in_and_extended_fact


def _apply_check_intermittent(
                                 gb,
                                 training_response,
                                 percent_zeros_threshold_for_intermittent
                             ):
    """

    :param gb:
    :param training_response:
    :param percent_zeros_threshold_for_intermittent:
    :return:
    """
    x = gb.apply(
                    _check_intermittent, 
                    training_response,
                    percent_zeros_threshold_for_intermittent
                )
    
    return x    


def _parallel_check_intermittent(
                                  pre_processed_data_set_list,
                                  key,
                                  training_response,
                                  percent_zeros_threshold_for_intermittent
                                ):
    """

    :param pre_processed_data_set_list:
    :param key:
    :param training_response:
    :param percent_zeros_threshold_for_intermittent:
    :return:
    """
    dask.config.set(scheduler='processes')
    
    task_list = []
    for i in range(len(pre_processed_data_set_list)):
        df_list_element = pre_processed_data_set_list[i]
        df = df_list_element[1]
        gb = df.groupby(key)
        fit_task = dask.delayed(_apply_check_intermittent)(
                                                               gb, 
                                                               training_response,
                                                               percent_zeros_threshold_for_intermittent
                                                          )
        task_list.append(fit_task)
            
    fit_task = dask.delayed(combine_into_df)(task_list)            
    filled_in_and_extended_fact  = fit_task.compute()
    
    return filled_in_and_extended_fact
    
def _shift_response(
                        x,
                        time_grain,
                        response_dict
                        ):
    """

    :param x:
    :param time_grain:
    :param response_dict:
    :return:
    """
    for k, response in response_dict.items():
        if time_grain == 'month':
            x['shifted_' + response] = x[response].shift(12).fillna(0)
            
        else:
            x['shifted_' + response] = x[response].shift(52).fillna(0)
        
    return x

def _add_benchmark(
                    filled_in_and_extended_fact,  
                    benchmark,
                    response_dict
                  ):
    """

    :param filled_in_and_extended_fact:
    :param benchmark:
    :param response_dict:
    :return:
    """
    if benchmark == 'last_year':
        for k, response in response_dict.items():        
            filled_in_and_extended_fact['benchmark'] = filled_in_and_extended_fact['shifted_' + response]
	
    elif benchmark == 'consensus':
        filled_in_and_extended_fact = filled_in_and_extended_fact.rename(columns = {'consensus':  'benchmark'})
    
    elif benchmark == 'stat':
        filled_in_and_extended_fact = filled_in_and_extended_fact.rename(columns = {'stat':  'benchmark'})
    
    filled_in_and_extended_fact['benchmark_type'] = benchmark    
    
    return filled_in_and_extended_fact

"""  
# appending holdout and forecast exog variables to the top-rigth of the matrix to avoid string comparison in fit_orchestrate
#def _append_exogs_to_df(x, exog_cols):
#    
#    x.reset_index(inplace=True)
#    holdout_exog_data = x[x.data_split == 'Holdout'][exog_cols].reset_index(drop=True)
#    forecast_exog_data = x[x.data_split == 'Forecast'][exog_cols].reset_index(drop=True)
#
#    x = pandas.concat([x, holdout_exog_data, forecast_exog_data], axis=1)
#    x.set_index(keys='index', inplace=True)
#	
#    return x
"""