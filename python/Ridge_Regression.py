from sklearn import linear_model
from Model_Parent import *
aq_df = get_air_quality_df()
ampg_df = get_auto_mpg_df()
ff_df = get_forest_fires_df()

forward_selection(linear_model.Ridge(), ff_df.iloc[:,:-1], ff_df.iloc[:,1])