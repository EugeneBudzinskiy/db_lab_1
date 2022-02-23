import os
import time

import psycopg2

from classes import FileArray
from classes import LoggerEngine
from config import *

from functions import convert_result
from functions import save_bar
from functions import save_csv
from functions import save_load_time

from queries import get_sql_length
from queries import get_sql_query
from queries import get_sql_scheme


def create_connection():
    connection = psycopg2.connect(
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        dbname=os.environ["PG_DB"],
        host=os.environ["PG_HOST"],
        port=os.environ["PG_PORT"]
    )
    return connection


def execute_query(connection, sql_query):
    buff = list()
    with connection.cursor() as curs:
        curs.execute(sql_query)
        if curs.description is not None:
            for row in curs:
                buff.append(tuple(map(lambda x: x.strip() if type(x) is str else x, row)))
    return None if len(buff) == 0 else buff


def import_csv(connection, table_name: str, file_array: FileArray, batch_size: int = 10000):
    with connection.cursor() as curs:
        row_count = execute_query(connection, sql_query=get_sql_length(TABLE_NAME))[0][0]
        file, data_len = file_array.get_data_batch(row_count, batch_size=batch_size)
        next(file)

        if data_len == 0:
            return False

        curs.copy_from(file, table=table_name, sep=file_array.delimiter, null=file_array.nan_value)
        return True


def log_row_count(connection, logger_engine: LoggerEngine):
    row_count = execute_query(connection, sql_query=get_sql_length(TABLE_NAME))[0][0]
    logger_engine.print_info(f'Rows already loaded: {row_count}')


def main():
    file_array = FileArray(
        cols_name=['id', 'region', 'status', 'score'],
        added_cols=['year'],
        delimiter=';',
        decimal='.',
        nan_value='null'
    )

    file_array.add_file(
        filepath='app/data/ZNO_2019.csv',
        cols_to_use=['OUTID', 'REGNAME', 'UkrTestStatus', 'UkrBall100'],
        delimiter=';',
        decimal=',',
        encoding='windows-1251',
        nan_value='null',
        value_to_add=[2019]
    )

    file_array.add_file(
        filepath='app/data/ZNO_2020.csv',
        cols_to_use=['OUTID', 'REGNAME', 'UkrTestStatus', 'UkrBall100'],
        delimiter=';',
        decimal=',',
        encoding='windows-1251',
        nan_value='null',
        value_to_add=[2020]
    )

    logger_engine = LoggerEngine()
    connection = create_connection()

    # Try to create table (if not exist)
    with connection as conn:
        execute_query(connection=conn, sql_query=get_sql_scheme(TABLE_NAME))

    # Continues upload the file and calculate time
    flag = True
    while flag:
        with connection as conn:
            start = time.time()
            log_row_count(connection=conn, logger_engine=logger_engine)
            flag = import_csv(conn, table_name=TABLE_NAME, file_array=file_array, batch_size=10000)
            end = time.time()
            save_load_time(TIME_RECORD_PATH, start=start, end=end)

    else:
        logger_engine.print_info('File fully loaded!')

    # Get result data (based on query)
    a_key, b_key = 2019, 2020
    title = 'Comparisons average score of Ukrainian Language test'
    y_label = 'Average score'

    with connection as conn:
        res = execute_query(connection=conn, sql_query=get_sql_query(TABLE_NAME))

    # Save CSV results
    header = ['region', 'year', 'avg_score']
    save_csv(data=res,
             header=header,
             filepath=RESULT_CSV_PATH,
             encoding=RESULT_CSV_ENCODING,
             delimiter=RESULT_CSV_DELIMITER)
    logger_engine.print_info('Result CSV file is saved!')

    # Save bar plot of results
    labels, a_values, b_values = \
        convert_result(result=res, cutout_word='область', a_key=a_key, b_key=b_key)
    save_bar(filepath=RESULT_PLOT_PATH,
             labels=labels,
             a_values=a_values,
             b_values=b_values,
             a_label=a_key,
             b_label=b_key,
             title=title,
             y_label=y_label,
             vertical_labels=True,
             size_inches=(12, 6))
    logger_engine.print_info('Bar plot is saved!')

    connection.close()
    time.sleep(1000)


if __name__ == '__main__':
    main()
