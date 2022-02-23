import logging

from io import StringIO
from sys import stdout


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
