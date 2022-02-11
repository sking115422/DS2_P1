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


def forward_selection(model, feature_df: pd.DataFrame, response_series: pd.Series):
    f_df = feature_df.copy(deep=True)
    features = []
    while not f_df.empty:
        best_f = None
        best_r2_bar = -1000
        best_r2_cv = -1000
        for f in f_df:
            temp_feats = features.append(f)
            model.fit(temp_feats, response_series)
            # GET R2 BAR and R2 CV
            # determine best_f
            ...
        # Print best_f's name and R^2
        features.append(best_f)
        f_df.drop(columns=[best_f], inplace=True)
    return


def backward_selection(df):
    return


def stepwise_selection(df):
    return
