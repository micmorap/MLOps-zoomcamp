import pandas as pd
from datetime import datetime
import batch
import pytest

def dt(hour, minute, second):

    return datetime(2022, 1, 18, hour, minute, second)

def prepare_data(df, categorical):    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


data = [
    (1, 1, dt(1, 2, 0), dt(1, 10, 0)),
    (1, 1, dt(1, 2, 0), dt(1, 10, 0)),
    (1, 2, dt(2, 2, 0), dt(2, 3, 0)),
    (2, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1))    
]

categorical = ['PULocationID', 'DOLocationID']
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

df_test = pd.DataFrame(data, columns=columns)
df_result_test = batch.prepare_data(df_test, categorical)

print(df_test)

print(df_result_test)