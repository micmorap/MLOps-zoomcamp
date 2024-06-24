import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from psycopg import sql


from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

create_table_statement = """
drop table if exists hw5_metrics_quantile;
create table hw5_metrics_quantile(
	date timestamp,
	fare_amount_quantile_day float
)
"""

raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create quantile table test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_quantile_postgresql(curr):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (datetime.datetime(2024, 3, 1, 0, 0))) &
        (raw_data.lpep_pickup_datetime < (datetime.datetime(2024, 4, 1, 0, 0)))
    ]

    current_data['date'] = current_data['lpep_pickup_datetime'].dt.date

    daily_medians = current_data.loc[current_data.lpep_pickup_datetime.between('2024-03-01', '2024-04-01', inclusive="left")].groupby('date')['fare_amount'].quantile(0.5)    

    for date, fare_amount_median in daily_medians.items():
        # Sentencia SQL de inserción
        insert_query = sql.SQL("INSERT INTO hw5_metrics_quantile (date, fare_amount_quantile_day) VALUES (%s, %s)")

        # Ejecutar la sentencia SQL de inserción
        curr.execute(insert_query, (date, fare_amount_median))

    #for index, row in current_data.iterrows():
    #    sql = "INSERT INTO {} (date, fare_amount_quantile_day) VALUES (%s, %s)".format('hw5_metrics_quantile')
    #    curr.execute(sql, (row['date'], row['fare_amount_quantile_day']))

@flow
def final_function():
    prep_db()
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        curr = conn.cursor()
        calculate_quantile_postgresql(curr)
        logging.info("data sent")


if __name__ == '__main__':
    final_function()

