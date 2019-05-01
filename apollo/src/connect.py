"""
# apollo demand forecast platform
# functions to connect to data stores to retrieve data
# @author: vikram govindan
# \params\data_store_credentials.json holds data store connection parameters
# replace the data_store_type to switch data stores
# turbodbc documentation: https://turbodbc.readthedocs.io/en/latest/index.html
# remember to close connections!
"""
from apollo.src import utils
import pyodbc
import pandas
import sqlalchemy
from azure.storage.blob import BlockBlobService

def _get_data_store_connection(data_store = 'sql_db', io_task = 'read'):
    """

    :param data_store:
    :param io_task:
    :return:
    """
    # private function to retrieve connection to data store

    # extend json file structure to accomodate different data sources
    
    data_store_connection_parameters = src.utils.read_params_in_from_json('data_store_credentials.json')[data_store]
    driver = data_store_connection_parameters['DRIVER']
    server = data_store_connection_parameters['SERVER']
    database = data_store_connection_parameters['DATABASE']
    username = data_store_connection_parameters['USERNAME']
    password = data_store_connection_parameters['PASSWORD']
    port = data_store_connection_parameters['PORT']
    
    if data_store == 'sql_db':
        
        if io_task == 'read':
    
            connection = pyodbc.connect(
                                            'Driver='+ driver
                                            +';Server='+ server
                                            +';Database='+ database
                                            +';Uid='+ username
                                            +';Pwd='+ password
                                            +';'
                                        )
            
        elif io_task == 'write':
            
            connection = sqlalchemy.create_engine(
                                                    'mssql+pyodbc://'
                                                  + username + ':' 
                                                  + password + '@'
                                                  + server + ':'
                                                  + port + '/'
                                                  + database + '?driver='
                                                  + driver
                                                 )

    else:

        connection = None

    return connection


def get_data(query, data_store = 'sql_db'):
    """

    :param query:
    :param data_store:
    :return:
    """
    # function to retrieve data from data store with connection
    
    if data_store == 'sql_db':
        
        connection = _get_data_store_connection(data_store)
        query_result = pandas.read_sql(query, connection)
        connection.close()
        
        return query_result



def write_data(dataframe, dB_table, if_exists = 'append', data_store = 'sql_db', schema = 'apollo'):
    """

    :param dataframe:
    :param dB_table:
    :param if_exists:
    :param data_store:
    :param schema:
    :return:
    """
    # function to retrieve data from data store with connection
    
    if data_store == 'sql_db':
    
        engine = _get_data_store_connection(data_store, io_task = 'write')
        dataframe.to_sql(
                            name = dB_table,
                            schema = schema,
                            con = engine,
                            if_exists = if_exists
                        )

    return None




def update_runtime_in_run_log(run_id, runtime_in_minutes, dB_table = 'RUN_LOG', data_store = 'sql_db', schema = 'apollo'):
    """

    :param run_id:
    :param runtime_in_minutes:
    :param dB_table:
    :param data_store:
    :param schema:
    :return:
    """
    if data_store == 'sql_db':
        
        connection = _get_data_store_connection(data_store)
        query = "UPDATE " + schema + '.' + dB_table + " SET runtime_in_minutes =  '" + str(runtime_in_minutes) + "', run_status = 'complete' " + " WHERE run_id = " + str(run_id)
        connection.execute(query)
        connection.commit()
        connection.close()
        
    return None



def _write_to_blob(output, blob_name):
    """

    :param output:
    :param blob_name:
    :return:
    """
    # function to retrieve data from data store with connection
    
    data_store = 'blob'
    data_store_connection_parameters = src.utils.read_params_in_from_json('data_store_credentials.json')[data_store]
    account_name = data_store_connection_parameters['ACCOUNT_NAME']
    account_key = data_store_connection_parameters['ACCOUNT_KEY']
    container_name = data_store_connection_parameters['CONTAINER_NAME']
    
    blobService = BlockBlobService(account_name=account_name, account_key=account_key)
    blobService.create_blob_from_text(container_name = container_name, blob_name = blob_name, text = output)

    return None


def _write_from_blob_to_azure_sql(schema, table, result_blob_name):
    """

    :param schema:
    :param table:
    :param result_blob_name:
    """
    data_store = 'blob'
    data_store_connection_parameters = src.utils.read_params_in_from_json('data_store_credentials.json')[data_store]
    external_source = data_store_connection_parameters['EXTERNAL_DATA_SOURCE']
    connection = _get_data_store_connection('sql_db')
    cursor = connection.cursor()
    query = "BULK INSERT " + schema + "." + table + " FROM '" + result_blob_name + "' WITH (DATA_SOURCE = '" + external_source + "', FORMAT = 'CSV', FIRSTROW = 2, ROWTERMINATOR = '0x0A', FIELDTERMINATOR = '\\t');"
    cursor.execute(query)
    connection.commit()
    connection.close()


def _bulk_write_data(
                        data,
                        main_info_logger,
                        run_config,
                        blob_name,
                        dB_table,
                        data_set_type,
                        schema = 'apollo'
                    ):
    """

    :param data:
    :param main_info_logger:
    :param run_config:
    :param blob_name:
    :param dB_table:
    :param data_set_type:
    :param schema:
    :return:
    """
    csv_for_blob = data.to_csv(
                                sep = '\t',
                                encoding = 'utf-8',
                                index=False
                               )

    main_info_logger.info('starting ' + data_set_type + ' write to blob...')
    _write_to_blob(csv_for_blob, blob_name)
    main_info_logger.info('write to blob complete...')
    
    main_info_logger.info('starting ' + data_set_type + ' write from blob to azure sql...')
    _write_from_blob_to_azure_sql(schema, dB_table, blob_name)
    main_info_logger.info('write to azure sql complete...')
    
    return None



def check_if_table_exists(schema, dB_table, data_store = 'sql_db'):
    """

    :param schema:
    :param dB_table:
    :param data_store:
    :return:
    """
    if data_store == 'sql_db':
        
        connection = _get_data_store_connection('sql_db')
        query = "IF OBJECT_ID (N'" + schema + "." + dB_table + "', N'U') IS NOT NULL SELECT 1 AS table_exists ELSE SELECT 0 AS table_exists;"
        query_result = pandas.read_sql(query, connection)
        connection.close()
    
    return query_result.table_exists.values[0]



def get_table_columns(schema, dB_table, data_store = 'sql_db'):
    """

    :param schema:
    :param dB_table:
    :param data_store:
    :return:
    """
    if data_store == 'sql_db':
        
        connection = _get_data_store_connection('sql_db')
        query = "SELECT TOP 1 * FROM " + schema + "." + dB_table
        query_result = pandas.read_sql(query, connection)
        connection.close()
    
    return list(query_result)



def run_query(query, data_store = 'sql_db'):
    """

    :param query:
    :param data_store:
    """
    if data_store == 'sql_db':
        
        connection = _get_data_store_connection(data_store)
        connection.execute(query)
        connection.commit()
        connection.close()


def _create_correct_table_if_not_exist(
                                        schema,
                                        dB_table,
                                        correct_field_list,
                                        ddl_query_file
                                      ):
    """

    :param schema:
    :param dB_table:
    :param correct_field_list:
    :param ddl_query_file:
    """
    does_result_table_exist = src.connect.check_if_table_exists(
                                                                    schema = schema,
                                                                    dB_table = dB_table
                                                               )
    
    create_result_table = True
    
    if correct_field_list is not None:
    
        if does_result_table_exist == 1:
            
            # check if result table columns are correct
            
            dB_table_fields = src.connect.get_table_columns(
                                                                schema = schema,
                                                                dB_table = dB_table
                                                           )
            create_result_table = dB_table_fields == correct_field_list
            
#            print(dB_table_fields)
#            print(correct_field_list)
#            print(create_result_table)
            
        else:
            
            create_result_table = False
    
    else:
        
        if does_result_table_exist == 0:
            
            create_result_table = False
        
    if not create_result_table:
        
        # create result table
        
        ddl_query = src.utils.read_in_file(ddl_query_file)
        src.connect.run_query(ddl_query)



def write_output(
                    data_set_type,
                    data_set,
                    run_config,
                    root,
                    main_info_logger
                ):
    """

    :param data_set_type:
    :param data_set:
    :param run_config:
    :param root:
    :param main_info_logger:
    """
    if data_set_type == 'result' or data_set_type == 'time_series_attribs':
        
        if data_set_type == 'result':
        
            blob = run_config['RESULT_BLOB_NAME']
            schema = run_config['RESULT_SCHEMA']
            table = run_config['RESULT_TABLE']
            query_filename = src.utils.read_params_in_from_json('queries.json')['CREATE_FACT_RESULT_TABLE']
        
        elif data_set_type == 'time_series_attribs':
            
            blob = run_config['TIME_SERIES_ATTRIBUTES_BLOB_NAME']
            schema = run_config['TIME_SERIES_ATTRIBUTES_SCHEMA']
            table = run_config['TIME_SERIES_ATTRIBUTES_TABLE']
            query_filename = src.utils.read_params_in_from_json('queries.json')['CREATE_TIME_SERIES_ATTRIBUTES']
        
        data_set = data_set.reindex_axis(sorted(data_set.columns), axis=1)
        data_set = data_set.drop('index', axis = 1, errors = 'ignore')
        
        ddl_query_file = root + '\\sql' + query_filename
        src.connect._create_correct_table_if_not_exist(
                                                        schema = schema,
                                                        dB_table = table,
                                                        correct_field_list = list(data_set),
                                                        ddl_query_file = ddl_query_file
                                                      )
        
        src.connect._bulk_write_data(
                                        data = data_set,
                                        main_info_logger = main_info_logger,
                                        run_config = run_config,
                                        blob_name = blob,
                                        schema = schema,
                                        data_set_type = data_set_type, 
                                        dB_table = table
                                    )
    
    if data_set_type == 'product_hierarchy':
        
        schema = run_config['PRODUCT_HIERARCHY_SCHEMA']
        table = run_config['PRODUCT_HIERARCHY_TABLE']
        query_filename = src.utils.read_params_in_from_json('queries.json')['CREATE_PRODUCT_HIERARCHY_TABLE']
        
        ddl_query_file = root + '\\sql' + query_filename
        src.connect._create_correct_table_if_not_exist(
                                                        schema = schema,
                                                        dB_table = table,
                                                        correct_field_list = None,
                                                        ddl_query_file = ddl_query_file
                                                     )
        
        ddl_query_file = root + '\\sql' + src.utils.read_params_in_from_json('queries.json')['PRODUCT_HIERARCHY']
        ddl_query = src.utils.read_in_file(ddl_query_file)
        src.connect.run_query(ddl_query)