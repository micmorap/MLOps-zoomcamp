#!/usr/bin/env python
# coding: utf-8


# get_ipython().system('pip freeze | grep scikit-learn')
# get_ipython().system('python -V')


import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# #### Q1. What's the standard deviation to March 2023 duration columns - Yellow trip dataset?
# stdev = y_pred.std()
# print(f"The standard deviation to {month} {year} duration predicted was {stdev}")

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    print(f"The predictions to {year}-{month:02d} has started!")    
    
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"    
    df = read_data(input_file)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)    
    
    df_predictions = pd.DataFrame(y_pred, columns=['predictions'])
    df_combined = pd.concat([df['ride_id'], df_predictions], axis=1)
    
    output_file = f"output_hw04/hw04_predictions_{year}_{month:02d}.parquet"
    
    df_combined.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    mean_predictions = y_pred.mean()

    print(f"The predictions parquet file to {year}-{month:02d} has been created and saved into output_hw04 folder!")

    msg = print(f"The mean predicted duration to {year}-{month:02d} was {mean_predictions}")

    
    return msg

if __name__ == '__main__':
    run()
