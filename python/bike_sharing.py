from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from Model_Parent import *

b_df = get_bike_sharing_df()

# Drops the Casual and Registered rental count columns along with total rented bikes since they are essentially the same metric
X = b_df.iloc[:, :-3]
# We are trying to predict total bikes rented, not just casual or registered users
y = b_df.iloc[:, -1]


def do_selections(model, X, y):
    forward_selection(model, X, y)
    backward_selection(model, X, y)
    stepwise_selection(model, X, y)


do_selections(linear_model.LinearRegression(), X, y)
do_selections(linear_model.Ridge(), X, y)

quad_reg = PolynomialFeatures(degree=2)
X_quad = quad_reg.fit_transform(X)
X_quad = pd.DataFrame(X_quad, columns=quad_reg.get_feature_names())
forward_selection(linear_model.LinearRegression(), X_quad, y)
