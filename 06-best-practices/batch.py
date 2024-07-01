#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import os


def prepare_data(df, categorical):    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def read_data(filename, categorical):
    
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)    
    
def save_data(filename, df):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
        print("Archivo parquet generado con éxito!")
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)
        print("Archivo parquet generado con éxito!")


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-mike-mora/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)



def main(year:int, month:int):

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    # Load the model and vectorizer
    try:
        with open('model.bin', 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print("Error: The model file 'model.bin' was not found.")
        exit(1)

    print(f"The predictions to {year}-{month:02d} has started!")    
    
    # input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"    
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)    
    
    df_predictions = pd.DataFrame(y_pred, columns=['predictions'])
    df_combined = pd.concat([df['ride_id'], df_predictions], axis=1)
    
    # output_file = f"output_hw06_files/hw06_predictions_{year}_{month:02d}.parquet"

    """
    df_combined.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    """

    save_data(output_file, df_combined)

    mean_predictions = y_pred.mean()

    print(f"The predictions parquet file to {year}-{month:02d} has been created and saved into output_hw06_files folder!")

    msg = print(f"The mean predicted duration to {year}-{month:02d} was {mean_predictions} minutes.")
    
    return msg


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])    
    
    main(year, month)
