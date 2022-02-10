import pandas as pd


def get_air_quality_df():
    df = pd.read_csv('cleaned_data/AirQualityUCI_fixed.csv')
    return df


def get_auto_mpg_df():
    df = pd.read_csv('cleaned_data/auto_mpg_fixed_cleaned.csv')
    return df


def get_forest_fires_df():
    df = pd.read_csv('cleaned_data/forestfires.csv')
    return df


def forward_selection(df):

    return


def backward_selection(df):
    return


def stepwise_selection(df):
    return
