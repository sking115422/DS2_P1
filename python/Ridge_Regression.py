from sklearn.linear_model import ridge_regression
from Model_Parent import *
aq_df = get_air_quality_df()
ampg_df = get_auto_mpg_df()
ff_df = get_forest_fires_df()

rr = ridge_regression()