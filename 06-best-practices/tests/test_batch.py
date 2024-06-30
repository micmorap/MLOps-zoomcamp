import pandas as pd
from datetime import datetime
import batch
import pytest

def dt(hour, minute, second):

    return datetime(2022, 1, 18, hour, minute, second)


def test_prepare_data():

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

    expected_data = [
        ('1', '1', dt(1, 2, 0), dt(1, 10, 0), 8.0),
        ('1', '1', dt(1, 2, 0), dt(1, 10, 0), 8.0),
        ('1', '2', dt(2, 2, 0), dt(2, 3, 0), 1.0)
    ]
    
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    df_expected = pd.DataFrame(expected_data, columns=expected_columns)
    
    pd.testing.assert_frame_equal(df_expected.reset_index(drop=True), df_result_test)



if __name__ == '__main__':
    pytest.main([__file__])
