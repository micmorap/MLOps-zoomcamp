import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def vectorizer_training_set(df: pd.DataFrame):
    
    dict_df = df['prepare_hw3']
    list_df = dict_df[0]
    
    df_locations = list_df[['PULocationID', 'DOLocationID']]
    data_dict = df_locations.to_dict(orient='records')

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(data_dict)
    y_train = list_df['duration'].values

    return X_train, y_train, dv
    
def train_model(X_train, y_train):
    model = LinearRegression()
    model_trained = model.fit(X_train, y_train)
    return model_trained


@transformer
def transform(df, *args, **kwargs):
    """
    """
    # Specify your transformation logic here
    X_train, y_train, dv = vectorizer_training_set(df)
    final_lr_model = train_model(X_train, y_train)

    print(f"The intercept of LR model is: {final_lr_model.intercept_}")
    return dv, final_lr_model