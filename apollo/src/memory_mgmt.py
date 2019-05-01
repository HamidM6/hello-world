# apollo demand forecast platform
# functions to determine attributes related to time series
# @author: vikram govindan

def get_df_memory_used(
                        df,
                        unit = 'mb'
                      ):
    
    memory_footprint_in_bytes = df.memory_usage(index = True, deep = True).sum()
    
    if unit == 'mb':
        
        memory_footprint = memory_footprint_in_bytes/1024/1024
        
    elif unit == 'gb':
        
        memory_footprint = memory_footprint_in_bytes/1024/1024/1024
    
    return memory_footprint

def optimize_raw_fact_memory(
                                raw_fact,
                                response_dict,
                                key
                            ):
    
    for k, response in response_dict.items():
        raw_fact[response] = raw_fact[response].astype('int32')
    raw_fact['parallel_partition_dense_rank'] = raw_fact['parallel_partition_dense_rank'].astype('int32')
    
    return raw_fact