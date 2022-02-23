import os
import time
import logging
import psycopg2

from matplotlib import pyplot as plt
from csv import writer as csv_writer
from io import StringIO
from sys import stdout
from config import *


class LoggerEngine:
    def __init__(self):
        self.log_level = logging.INFO

        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        self.log = logging.getLogger(__name__)
        self.log.setLevel(self.log_level)

        handler = logging.StreamHandler(stdout)
        handler.setLevel(self.log_level)
        handler.setFormatter(log_format)
        self.log.addHandler(handler)

    def print_info(self, message: str):
        self.log.info(message)


class FileArray:
    def __init__(self,
                 cols_name: list,
                 delimiter: str = ',',
                 decimal: str = '.',
                 nan_value: str = 'null',
                 added_cols: list = None):
        self.instructions = dict()

        self.cols_name = cols_name
        self.added_cols = added_cols

        self.delimiter = delimiter
        self.decimal = decimal
        self.nan_value = nan_value

    def add_file(self,
                 filepath: str,
                 cols_to_use: list,
                 delimiter: str = ',',
                 decimal: str = '.',
                 encoding: str = 'utf-8',
                 nan_value: str = 'null',
                 value_to_add: list = None):

        if filepath in self.instructions:
            raise Exception(f'Error! File with path `{filepath}` already added!')

        if len(cols_to_use) != len(self.cols_name):
            raise Exception(f'Error! `cols_to_use` and `cols_new_names` have different lengths')

        self.instructions[filepath] = dict()
        self.instructions[filepath]['delimiter'] = delimiter
        self.instructions[filepath]['decimal'] = decimal

        self.instructions[filepath]['encoding'] = encoding
        self.instructions[filepath]['cols_to_use'] = cols_to_use
        self.instructions[filepath]['nan_value'] = nan_value

        if self.added_cols is None:
            if value_to_add is not None:
                raise Exception(f'Error! Doesnt except value in `value_to_add`')

            self.instructions[filepath]['value_to_add'] = value_to_add
        else:
            if len(self.added_cols) != len(value_to_add):
                raise Exception(f'Error! `added_cols` and `value_to_add` have different lengths')

            value_to_add_full = [None for _ in self.added_cols] if value_to_add is None else value_to_add
            self.instructions[filepath]['value_to_add'] = value_to_add_full

    def _str_processor(self, filepath: str, row_data: str, header_data: str):
        data = [el.replace('"', '') for el in row_data.split(self.instructions[filepath]['delimiter'])]
        header = [el.replace('"', '') for el in header_data.split(self.instructions[filepath]['delimiter'])]

        buff = list()
        for el in self.instructions[filepath]['cols_to_use']:
            idx = header.index(el)
            buff.append(data[idx])

        for i in range(len(buff)):
            buff[i] = buff[i].replace(self.instructions[filepath]['decimal'], self.decimal)
            buff[i] = buff[i].replace(self.instructions[filepath]['nan_value'], self.nan_value)

        for el in self.instructions[filepath]['value_to_add']:
            buff.append(str(el))

        return self.delimiter.join(buff)

    def _file_processor(self, buffer: list, filepath: str, start_from: int, batch_size: int):
        with open(filepath, 'r', encoding=self.instructions[filepath]['encoding']) as f:
            header = f.readline()
            k = 0
            while True:
                row = f.readline()
                if row == '':
                    return False, k

                if k >= start_from:
                    if k >= start_from + batch_size:
                        return True, 0

                    buffer.append(self._str_processor(filepath=filepath, row_data=row, header_data=header))

                k += 1

    def get_data_batch(self, start_from, batch_size: int = 100):
        result = list()
        start = start_from
        for key in self.instructions.keys():
            is_end, k_number = \
                self._file_processor(buffer=result, filepath=key, start_from=start, batch_size=batch_size)
            start -= k_number

            if is_end:
                break

        file_header = self.delimiter.join(self.cols_name + self.added_cols)
        return StringIO('\n'.join([file_header] + result)), len(result)


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


def get_sql_scheme(table_name: str):
    query = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        ID CHAR(64) NOT NULL PRIMARY KEY,
        REGION CHAR(256) NULL,
        STATUS CHAR(100) NULL,
        SCORE REAL NULL,
        YEAR INTEGER NULL
    );'''
    return query


def get_sql_query(table_name: str):
    query = f'''
    SELECT region, year, AVG(score) AS avg_score FROM {table_name}
        WHERE status = 'Зараховано' GROUP BY region, year;
    '''
    return query


def get_sql_length(table_name: str):
    return f'''SELECT COUNT(*) FROM {table_name};'''


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


def save_load_time(filepath: str, start: float, end: float):
    template = 'Load time (s): {}'
    try:
        with open(filepath, 'r') as f:
            file_data = f.read()
            time_value = float(file_data.split(':')[-1].strip())

        with open(filepath, 'w') as f:
            f.write(template.format(time_value + (end - start)))

    except FileNotFoundError:
        with open(filepath, 'w') as f:
            f.write(template.format(end - start))


def save_bar(filepath: str,
             labels: list,
             a_values: list,
             b_values: list,
             a_label: str = 'A values',
             b_label: str = 'B values',
             title: str = 'Title',
             y_label: str = 'y label',
             vertical_labels: bool = False,
             size_inches: tuple = None,
             bar_width: float = .4):

    x = [x for x in range(len(labels))]
    fig, ax = plt.subplots()
    if size_inches is not None:
        fig.set_size_inches(*size_inches)

    ax.bar([x - bar_width / 2 for x in x], a_values, bar_width, label=a_label)
    ax.bar([x + bar_width / 2 for x in x], b_values, bar_width, label=b_label)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    rotation = 'vertical' if vertical_labels else 'horizontal'
    ax.set_xticklabels(labels, rotation=rotation)
    ax.legend()

    fig.tight_layout()
    plt.savefig(filepath)


def convert_result(result: list, cutout_word: str, a_key, b_key):
    res_dict = dict()
    for el in result:
        region, year, score = el
        region = region.replace(cutout_word, '').strip()
        if region not in res_dict:
            res_dict[region] = dict()
        res_dict[region][year] = score

    labels = list(res_dict.keys())
    a_values, b_values = list(), list()
    for el in labels:
        a_values.append(res_dict[el][a_key])
        b_values.append(res_dict[el][b_key])

    return labels, a_values, b_values


def save_csv(data: list, header: list, filepath: str, encoding: str, delimiter: str):
    with open(filepath, 'w', encoding=encoding) as f:
        file_writer = csv_writer(f, delimiter=delimiter)
        file_writer.writerow(list(map(str, header)))
        for el in data:
            file_writer.writerow(list(map(str, el)))


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
