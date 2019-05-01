"""
|  apollo demand forecast platform
|  @author: vikram govindan
|  apollo main
"""

import datetime

# reload logging since ipython configures a logging handler in advance

import logging
from imp import reload
reload(logging)

import gc

import os
import pandas
import random
import time

# import custom function modules
from apollo.src import connect
from apollo.src import constants
from apollo.src import cleanse_history
#from apollo.fit import fit.ensemble 
from apollo.src import fit_orchestrate
from apollo.src import model_selection
from apollo.src import post_process
from apollo.src import pre_process
#from apollo.src import time_series_attribs
from apollo.src import utils

# suppress ALL warnings
# uncomment to review all warnings
import warnings
warnings.filterwarnings("ignore")

t_start = time.time()

###################################### setup logging ######################################

root = os.path.dirname(os.getcwd())
main_info_log_filename = root + '\\log\\main.log'
model_fit_error_log_filename = root + '\\log\\error.log'
post_process_error_log_filename = root + '\\log\\post_process.log'
runtime_stats_info_log_filename = root + '\\log\\runtime_stats.log'
utils.setup_logger('main_info_logger', main_info_log_filename, level = logging.INFO)
utils.setup_logger('model_fit_error_logger', model_fit_error_log_filename, level = logging.ERROR)
utils.setup_logger('post_process_error_logger', post_process_error_log_filename, level = logging.ERROR)
utils.setup_logger('runtime_stats_info_logger', runtime_stats_info_log_filename, level = logging.INFO)
main_info_logger = logging.getLogger('main_info_logger')
model_fit_error_logger = logging.getLogger('model_fit_error_logger')
post_process_error_logger = logging.getLogger('post_process_error_logger')
runtime_stats_info_logger = logging.getLogger('runtime_stats_info_logger')

###################################### read in parameters for model and run configuration ###########

run_config = utils.read_params_in_from_json('run_config.json')
model_param_config = utils.read_params_in_from_json('model_param_config.json')

key = run_config['FORECAST_BY_DIMENSIONS']
data_filters = run_config['DATA_FILTERS']
response_dict = run_config['RESPONSES']
response_mapping = run_config['RESPONSE_MAPPING']
training_response = response_dict[response_mapping['training']]
holdout_response = response_dict[response_mapping['holdout']]
forecast_response = response_dict[response_mapping['forecast']]
cap_outliers = eval(model_param_config['CAP_OUTLIERS'])
benchmark = run_config['BENCHMARK']
time_grain = run_config['FORECAST_TIME_GRAIN']
pass_through_key_attributes = run_config['PASS_THROUGH_KEY_ATTRIBUTES']
pre_process_execution_path = run_config['EXECUTION_PATH']['pre_process']
fit_execution_path = run_config['EXECUTION_PATH']['fit']
post_process_execution_path = run_config['EXECUTION_PATH']['post_process']
parallel_partition_key = run_config['PARALLEL_PARTITION_KEY']
parallel_partitions = run_config['PARALLEL_PARTITIONS']
benchmark = run_config['BENCHMARK']

epsilon = constants.constants().EPSILON
model_list = list(set(model_param_config['MODELING_TECHNIQUE_LIST']))
model_time_series_required_length = model_param_config['MODEL_TIME_SERIES_REQUIRED_LENGTH']
ensemble = eval(model_param_config['ENSEMBLE'])
ensemble_model_list = model_param_config['ENSEMBLE_MODEL_LIST']

###################################### record run attributes #####################################

# generate random 32 bit run id
run_id = random.getrandbits(32)
data_scientist = run_config['DATA_SCIENTIST']
run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

run_log = pandas.DataFrame(
                                {
                                    'run_id':                       [run_id],
                                    'data_scientist':               [data_scientist],
                                    'run_date':                     [run_date],
                                    'training_response':            [training_response],
                                    'holdout_response':             [holdout_response],
                                    'forecast_response':            [forecast_response],
                                    'time_grain':                   [time_grain],
                                    'key':                          [', '.join(key)],
                                    'fit_execution_path':           [fit_execution_path],                                                                       
                                    'run_status':                   ['in_progress'],
                                    'runtime_in_minutes':           [None],
                                    'post_process_execution_path':  [post_process_execution_path],
                                    'pre_process_execution_path':   [pre_process_execution_path]
                                }                                        
                         )

connect.write_data(
                          dataframe = run_log,
                          dB_table = 'RUN_LOG'                                                
                      )

###################################### run attributes #####################################

main_info_logger.info('run: ' + str(run_id) + ' of ' + 'apollo kicking off ' + ' at the ' + ' '.join(key) + ' ' + time_grain + ' level...')

###################################### read in data ######################################

(raw_fact, event_calendar) = pre_process.read_in_data(
                                                            run_config = run_config,
                                                            main_info_logger = main_info_logger,
                                                            time_grain = time_grain,
                                                            key = key,
                                                            data_filters = data_filters,
                                                            response_dict = response_dict,
                                                            training_response = training_response,
                                                            holdout_response = holdout_response,
                                                            forecast_response = forecast_response,
                                                            pass_through_key_attributes = pass_through_key_attributes,
                                                            benchmark = benchmark
                                                          )

###################################### pre process #######################################

(filled_in_and_extended_fact, pivot_event_calendar, time_series_attribs, holidays, exog_cols) =  pre_process.prepare_for_model_fit(
                                                                                                                                         raw_fact = raw_fact,
                                                                                                                                         model_param_config = model_param_config,
                                                                                                                                         main_info_logger = main_info_logger,
                                                                                                                                         event_calendar = event_calendar,
                                                                                                                                         benchmark = benchmark,
                                                                                                                                         key = key,
                                                                                                                                         time_grain = time_grain,
                                                                                                                                         training_response = training_response,
                                                                                                                                         forecast_response = forecast_response,
                                                                                                                                         response_dict = response_dict,
                                                                                                                                         run_id = run_id,
                                                                                                                                         pass_through_key_attributes = pass_through_key_attributes,
                                                                                                                                         pre_process_execution_path = pre_process_execution_path,
                                                                                                                                         parallel_partition_key = parallel_partition_key,
                                                                                                                                         epsilon = epsilon,
                                                                                                                                         cap_outliers = cap_outliers
                                                                                                                                     )

no_data = filled_in_and_extended_fact.empty
if no_data is False:  
    
    training = filled_in_and_extended_fact[filled_in_and_extended_fact.data_split == 'Training'].groupby(key)
    holdout = filled_in_and_extended_fact[filled_in_and_extended_fact.data_split == 'Holdout'].groupby(key)
    complete = filled_in_and_extended_fact[filled_in_and_extended_fact.data_split != 'Forecast'].groupby(key)
    forecast = filled_in_and_extended_fact[filled_in_and_extended_fact.data_split == 'Forecast'].groupby(key)
    training_keys = training.groups
    n_keys = len(training_keys)
    pre_processed_dict = {}
    for gbkey in training_keys:
        pre_processed_dict[gbkey] = {
                                        'training':         training.get_group(gbkey),
                                        'holdout':          holdout.get_group(gbkey),
                                        'complete':         complete.get_group(gbkey),
                                        'forecast':         forecast.get_group(gbkey)
                                    }
    
################################### clean up #############################################

    gc.collect()
    del(
        #raw_fact,
        training,
        holdout,
        complete,
        forecast
        )
    
################################### non parallel ########################################
    
    if fit_execution_path == 'non_parallel':          
        
        fit_results_training = fit_orchestrate.non_parallel_model_fit(
                                                                             main_info_logger = main_info_logger,
                                                                             data_dict = pre_processed_dict,
                                                                             model_list = model_list,
                                                                             time_grain = time_grain,
                                                                             training_response = training_response,
                                                                             forecast_response = holdout_response,
                                                                             key = key,
                                                                             model_fit_error_logger = model_fit_error_logger,
                                                                             model_time_series_required_length = model_time_series_required_length,
                                                                             epsilon = epsilon,
                                                                             pass_through_key_attributes = pass_through_key_attributes,
                                                                             benchmark = benchmark,
                                                                             holidays = holidays,
                                                                             exog_cols = exog_cols,
                                                                             fit_type = 'training',
                                                                             n_keys = str(n_keys)
                                                                        )
        
        
        fit_results_training = fit_orchestrate.non_parallel_ensemble_fit(
                                                                                main_info_logger = main_info_logger,
                                                                                n_keys = str(n_keys),
                                                                                fit_results_dict = fit_results_training,
                                                                                ensemble_model_list = ensemble_model_list,
                                                                                fit_type = 'training',
                                                                                ensemble = ensemble
                                                                            )
        
        (pre_processed_dict, fit_results_training) = fit_orchestrate.non_parallel_pick_and_pass_winning_and_best_model(
                                                                                                                                fit_results_training = fit_results_training,
                                                                                                                                model_list = model_list,
                                                                                                                                ensemble = ensemble,
                                                                                                                                post_process_execution_path = post_process_execution_path,
                                                                                                                                main_info_logger = main_info_logger,
                                                                                                                                n_keys = str(n_keys),
                                                                                                                                epsilon = epsilon,
                                                                                                                                pre_processed_dict = pre_processed_dict
                                                                                                                          )      
        
        fit_results_complete = fit_orchestrate.non_parallel_model_fit(
                                                                             main_info_logger = main_info_logger,
                                                                             data_dict = pre_processed_dict,
                                                                             model_list = model_list,
                                                                             time_grain = time_grain,
                                                                             training_response = forecast_response,
                                                                             forecast_response = forecast_response,
                                                                             key = key,
                                                                             model_fit_error_logger = model_fit_error_logger,
                                                                             model_time_series_required_length = model_time_series_required_length,
                                                                             epsilon = epsilon,
                                                                             pass_through_key_attributes = pass_through_key_attributes,
                                                                             benchmark = benchmark,
                                                                             holidays = holidays,
                                                                             exog_cols = exog_cols,
                                                                             fit_type = 'complete',
                                                                             n_keys = str(n_keys),
                                                                             ensemble_model_list = ensemble_model_list
                                                                        )
        
        
        fit_results_complete = fit_orchestrate.non_parallel_ensemble_fit(
                                                                                main_info_logger = main_info_logger,
                                                                                n_keys = str(n_keys),
                                                                                fit_results_dict = fit_results_complete,
                                                                                ensemble_model_list = ensemble_model_list,
                                                                                fit_type = 'complete',
                                                                                ensemble = ensemble
                                                                            ) 
    
####################################### parallel ########################################
    
    if fit_execution_path == 'parallel':
        
        fit_results_training = fit_orchestrate.parallel_model_fit(
                                                                            root = root,
                                                                            run_id = run_id,
                                                                            pre_processed_dict = pre_processed_dict,
                                                                            parallel_partitions = parallel_partitions,
                                                                            model_list = model_list,
                                                                            time_grain = time_grain,
                                                                            training_response = training_response,
                                                                            forecast_response = holdout_response,
                                                                            key = key,
                                                                            model_fit_error_logger = model_fit_error_logger,
                                                                            model_time_series_required_length = model_time_series_required_length,
                                                                            epsilon = epsilon,
                                                                            pass_through_key_attributes = pass_through_key_attributes,
                                                                            benchmark = benchmark,
                                                                            holidays = holidays,
                                                                            exog_cols = exog_cols,
                                                                            fit_type = 'training',
                                                                            main_info_logger = main_info_logger,
                                                                            n_keys = str(n_keys)
                                                                     )        
        
        
        fit_results_training = fit_orchestrate.parallel_fit_ensemble(
                                                                            root = root,
                                                                            run_id = run_id,
                                                                            fit_results_dict = fit_results_training,
                                                                            fit_type = 'training',
                                                                            parallel_partitions = parallel_partitions,
                                                                            ensemble_model_list = ensemble_model_list,
                                                                            main_info_logger = main_info_logger,
                                                                            n_keys = str(n_keys)
                                                                        )
        
        (pre_processed_dict, fit_results_training) = fit_orchestrate.parallel_pick_and_pass_winning_and_best_model(
                                                                                                                            fit_results_dict = fit_results_training,
                                                                                                                            parallel_partitions = parallel_partitions,
                                                                                                                            model_list = model_list,
                                                                                                                            ensemble = ensemble,
                                                                                                                            post_process_execution_path = post_process_execution_path,
                                                                                                                            main_info_logger = main_info_logger,
                                                                                                                            n_keys = str(n_keys),
                                                                                                                            epsilon = epsilon,
                                                                                                                            pre_processed_dict = pre_processed_dict
                                                                                                                      )
        
        fit_results_complete = fit_orchestrate.parallel_model_fit(
                                                                            root = root,
                                                                            run_id = run_id,
                                                                            pre_processed_dict = pre_processed_dict,
                                                                            parallel_partitions = parallel_partitions,
                                                                            model_list = model_list,
                                                                            time_grain = time_grain,
                                                                            training_response = forecast_response,
                                                                            forecast_response = forecast_response,
                                                                            key = key,
                                                                            model_fit_error_logger = model_fit_error_logger,
                                                                            model_time_series_required_length = model_time_series_required_length,
                                                                            epsilon = epsilon,
                                                                            pass_through_key_attributes = pass_through_key_attributes,
                                                                            benchmark = benchmark,
                                                                            holidays = holidays,
                                                                            exog_cols = exog_cols,
                                                                            fit_type = 'complete',
                                                                            main_info_logger = main_info_logger,
                                                                            n_keys = str(n_keys),
                                                                            ensemble_model_list = ensemble_model_list
                                                                     )
        
        fit_results_complete = fit_orchestrate.parallel_fit_ensemble(
                                                                            root = root,
                                                                            run_id = run_id,
                                                                            fit_results_dict = fit_results_complete,
                                                                            fit_type = 'complete',
                                                                            parallel_partitions = parallel_partitions,
                                                                            ensemble_model_list = ensemble_model_list,
                                                                            main_info_logger = main_info_logger,
                                                                            n_keys = str(n_keys)
                                                                        )
    
    
    main_info_logger.info('model build complete...')
    
    
####################################### post process #####################################
    
    t_pp_start = time.time()    
    fit_results = post_process.combine_fit_result_dicts(
                                                             fit_results_training = fit_results_training,
                                                             fit_results_complete = fit_results_complete
                                                           )
    
    result = post_process.compile_results(
                                                    fit_results = fit_results,
                                                    filled_in_and_extended_fact = filled_in_and_extended_fact,
                                                    post_process_error_logger = post_process_error_logger,
                                                    runtime_stats_info_logger = runtime_stats_info_logger,
                                                    root = root,
                                                    model_list = model_list,
                                                    pivot_event_calendar = pivot_event_calendar,
                                                    run_id = run_id,
                                                    response_dict = response_dict,
                                                    training_response = training_response,
                                                    holdout_response = holdout_response,
                                                    forecast_response = forecast_response,
                                                    key = key,
                                                    run_config = run_config,
                                                    time_grain = time_grain,
                                                    epsilon = epsilon,
                                                    post_process_execution_path = post_process_execution_path,
                                                    parallel_partition_key = parallel_partition_key,
                                                    parallel_partitions = parallel_partitions,
                                                    time_series_attribs = time_series_attribs
                                            )
    
    t_pp_end = time.time()    
    main_info_logger.info('post process runtime = ' + str((t_pp_end - t_pp_start)/60) + ' minutes')

####################################### write ####################################
    
    
    connect.write_output(
                                data_set_type = 'result',
                                data_set = result,
                                run_config = run_config,
                                root = root,
                                main_info_logger = main_info_logger
                            )
    
#    connect.write_output(
#                                data_set_type = 'time_series_attribs',
#                                data_set = time_series_attribs,
#                                run_config = run_config,
#                                root = root,
#                                main_info_logger = main_info_logger
#                            )
    
#    connect.write_output(
#                                data_set_type = 'product_hierarchy',
#                                data_set = None,
#                                run_config = run_config,
#                                root = root,
#                                main_info_logger = main_info_logger
#                            )
 
    
    t_end = time.time()
    runtime_in_minutes = (t_end - t_start)/60
    main_info_logger.info('total runtime = ' + str(runtime_in_minutes) + ' minutes')
    
    
############ run log ##############
    
    connect.update_runtime_in_run_log(
                                            runtime_in_minutes = runtime_in_minutes,
                                            run_id = run_id
                                        )
    
    main_info_logger.info('all set.')