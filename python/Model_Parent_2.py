

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from RegscorePy import * 
import statsmodels.api as sm

# library I got aic and bic from is listed below. Underlying calculations are shown there.
# https://pypi.org/project/RegscorePy/




def showFit(x_values, y_test, y_pred):
    plt.scatter(x_values, y_test, label="y_test", color="black")
    plt.plot(x_values, y_pred, label="y_pred", color="blue")
    plt.legend()
    plt.show()




def calc_r2_bar(m, n, r2):
        
    #m = number of data points
    #n = number of features
    
    dfr = n - 1
    df = m - n
    
    rdf = (dfr + df) / df           #ratio of total degrees of freedom to degrees of freedom for error 
    r2_bar = 1 - rdf * (1 - r2)
    
    return r2_bar




def splitData(X, y, num_splits):

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=1)
    kf.get_n_splits(X_arr, y_arr)
    
    ret_list = []

    for train_index, test_index in kf.split(X_arr, y_arr):
        X_train, X_test, y_train, y_test = X_arr[train_index], X_arr[test_index], y_arr[train_index], y_arr[test_index]
        
        ret_list.append([X_train, X_test, y_train, y_test])
        
    return ret_list




def calc_p_values(model, X, y, bool, name):
    # Statsmodels.OLS requires us to add a constant.
    x = sm.add_constant(X)
    model = sm.OLS(y , x)
    results = model.fit()
    
    print()
    print(name + " SELECTION REPORT:")
    print()
    print(results.summary())
    
    df_pval = pd.DataFrame()
    df_pval['pval'] = results.pvalues
    df_pval.drop(index=df_pval.index[0], axis=0, inplace=True)
    df_pval['feat_name'] = df_pval.index
    df_pval.reset_index(drop=True, inplace=True)
    df_pval['index'] = df_pval.index
    df_pval = df_pval[['index', 'feat_name', 'pval']]
    df_pval = df_pval.sort_values('pval', ascending=bool)
    
    return df_pval




def forwardSelection(model, X, y):

    x_tmp = pd.DataFrame()
    
    # X.shape[1]
    
    r2_cv_list_final = []
    r2_bar_list_final = []
    aic_list_final = []
    bic_list_final = []
    
    #True = sorting ascending order => forward selection
    df_pval = calc_p_values(model, X, y, True, "FORWARD")
    
    add_order_list = df_pval['index'].to_list()
    feature_names_list = df_pval['feat_name'].to_list()
    
    num_feat = 0

    for each in add_order_list:
        
        num_feat = num_feat + 1
        
        x_tmp = pd.concat([x_tmp, X.iloc[:, each]], axis=1)
        
        ret_list = splitData(x_tmp, y, 5)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            
            r2_bar_list.append(calc_r2_bar(len(y), num_feat, r2))
            
            aic_val_list.append(aic.aic(y_test, y_pred, num_feat))
            
            bic_val_list.append(bic.bic(y_test, y_pred, num_feat))
            
        
        r2_cv_list_final.append(np.average(r2_list))
        r2_bar_list_final.append(np.average(r2_bar_list))
        aic_val_list.append(np.average(aic_val_list))
        bic_val_list.append(np.average(aic_val_list))
        
    feature_list = range(1, X.shape[1] + 1)
    
    df_final = pd.DataFrame()
    df_final['Num_Features'] = feature_list
    df_final['r2_cv'] = r2_cv_list_final
    df_final['r2_bar'] = r2_bar_list_final
    df_final['aic'] = aic_val_list
    df_final['bic'] = bic_val_list
    
    print()
    print("FORWARD SELECTION SUMMARY TABLE:")
    print()
    print("Features In Order Added:", feature_names_list)

    t = pt(['Num_Features', 'r2_cv', 'r2_bar', 'AIC', 'BIC'])
    for row in range(0, df_final.shape[0]):
        t.add_row(df_final.iloc[row, :].to_list())
    print(t)
    
    
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Forward Selection Graphical Summary')    
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(feature_list, r2_cv_list_final, label="r2_cv", color="green")
    ax1.plot(feature_list, r2_bar_list_final, label="r2_bar", color="red")
    ax1.legend()
        
    ax2.plot(feature_list, aic_val_list, label="aic", color="blue")
    ax2.plot(feature_list, bic_val_list, label="bic", color="orange")
    ax2.legend()
    
    fig.show()




def backwardSelection(model, X, y):

    x_tmp = pd.DataFrame()

    # X.shape[1]

    r2_cv_list_final = []
    r2_bar_list_final = []
    aic_list_final = []
    bic_list_final = []

    #False = sorting descending order => backward selection
    df_pval = calc_p_values(model, X, y, False, "BACKWARD")

    rem_order_list = df_pval['index'].to_list()
    feature_names_list = df_pval['feat_name'].to_list()

    num_feat = X.shape[1]
    
    x_tmp = X

    for each in feature_names_list:
        
        ret_list = splitData(x_tmp, y, 5)
        
        r2_list = []
        r2_bar_list = []
        aic_val_list = []
        bic_val_list = []
        
        for fold in ret_list:
            
            X_train = fold[0]
            X_test = fold[1]
            y_train = fold[2]
            y_test = fold[3]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)
            
            r2_bar_list.append(calc_r2_bar(len(y), num_feat, r2))
            
            aic_val_list.append(aic.aic(y_test, y_pred, num_feat))
            
            bic_val_list.append(bic.bic(y_test, y_pred, num_feat))
            
        
        r2_cv_list_final.append(np.average(r2_list))
        r2_bar_list_final.append(np.average(r2_bar_list))
        aic_val_list.append(np.average(aic_val_list))
        bic_val_list.append(np.average(aic_val_list))
    
        x_tmp = x_tmp.drop(each, axis=1)
    
        num_feat = num_feat - 1
    
    feature_list = range(1, X.shape[1] + 1)
    feature_list = list(feature_list)
    feature_list = feature_list[::-1]

    df_final = pd.DataFrame()
    df_final['Num_Features'] = feature_list
    df_final['r2_cv'] = r2_cv_list_final
    df_final['r2_bar'] = r2_bar_list_final
    df_final['aic'] = aic_val_list
    df_final['bic'] = bic_val_list

    print()
    print("BACKWARD SELECTION SUMMARY TABLE:")
    print()
    print("Features In Order Removed:", feature_names_list)

    t = pt(['Num_Features', 'r2_cv', 'r2_bar', 'AIC', 'BIC'])
    for row in range(0, df_final.shape[0]):
        t.add_row(df_final.iloc[row, :].to_list())
    print(t)


    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Backward Selection Graphical Summary')    

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(feature_list, r2_cv_list_final, label="r2_cv", color="green")
    ax1.plot(feature_list, r2_bar_list_final, label="r2_bar", color="red")
    ax1.legend()
        
    ax2.plot(feature_list, aic_val_list, label="aic", color="blue")
    ax2.plot(feature_list, bic_val_list, label="bic", color="orange")
    ax2.legend()

    fig.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
### More Concise Method For CV Testing
    
# scoring = ['r2', 'neg_mean_squared_error' ]

# import sklearn
# print(sorted(sklearn.metrics.SCORERS.keys()))

# len(df.columns)

# x_tmp = pd.DataFrame()
# results = []

# for each in range(0, X.shape[1]):
    
#     x_tmp = pd.concat([x_tmp, df.iloc[:, each]], axis=1)
    
#     res = cross_validate(model, x_tmp, y, scoring=scoring, cv=10)  # 80-20 split
    
#     results.append(res)