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
