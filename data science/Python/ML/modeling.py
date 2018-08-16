# coding: utf-8
# modeling module
# all functions ending with _by_masterSKU() train a model on the entire training data
# and return individual masterSKU RMSE and SMAPE on test data
import pandas as pd
import numpy as np
import datetime
import copy
import pyodbc
import sys
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox, levene, bartlett
from scipy import linalg


def main():
    pass
if __name__ == "__main__":
    main()

##########################################################################################
###################### Constant ##########################################################
##########################################################################################

# dummy encoded masterSKU in train and test data
# list index = masterSKU encoding in train/test
# eg. Flip 12 = 0, Flip 18 = 1 and so on
ENCODED_MASTERSKUS = ['flip 12','flip 18','flip 8','Hopper 2.0 and Hopper 30','R colsters',
    'R10 lowball','R18 bottle','R20 tumbler','R26 bottle','R30 tumbler','R36 bottle',
    'Roadie 20','Tank 45-85','Tundra 105-350','Tundra 35','Tundra 45','Tundra 65','Tundra 75']

def find_product_category(masterSKU_id):
    op = None
    if masterSKU_id in range(4):
        op = 2
    elif masterSKU_id in range(4,11):
        op = 1
    elif masterSKU_id in range(11,18):
        op = 0
    return op

##########################################################################################
####################### split training and testing #######################################
##########################################################################################

def preprocess_data(data, frac, feature_list):
    
    """ convert data from pd.DataFrame to float64 nparrays """
    
    if frac != 1:
        data = data.sample(frac=frac, replace=False)
    
    X = data[feature_list].astype('float64',copy=False).values
    Y = data['monthly_sum_order_qty'].astype('float64',copy=False).values.ravel()
    
    return X, Y


def split_train_test(train, test, frac=1, masterSKU=None, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):
    
    """ return date-split train/test in np.float64 arrays """
    """ allows selection by masterSKU and/or sampling partial dataset for parameter tuning """
    """ for prediction, pass all data as training data, and set test=None """
    
    if masterSKU is not None:
        train = train.loc[(train['MasterSKU'] == masterSKU)]
        test = test.loc[(test['MasterSKU'] == masterSKU)]
        
    X_train, Y_train = preprocess_data(train, frac=frac, feature_list=feature_list)
    op = X_train, Y_train
    
    if test is not None:
        
        X_test, Y_test = preprocess_data(test, frac=frac, feature_list=feature_list)
        
        if masterSKU is None:
            op = X_train, X_test, Y_train, Y_test
            print("training: {0:.2%}".format(len(X_train)/(len(train)+len(test))))
            print("features:", feature_list)
        else:
            op = X_test, Y_test
            
    return op


##########################################################################################
################### evaluation & stats helper functions ##################################
##########################################################################################

def visualize_Y_distribution(Y, boxcox_transform=False):    
    original = Y.shape
    if boxcox_transform:
        Y, _ = boxcox(Y)
        t = "Response vector distribution after Box-Cox transformation"
        if Y.shape != original:
            raise ValueError("Check scipy.stats.BoxCox()")
    t = "Original response vector distribution"
    plt.hist(Y)
    plt.title(t)
    #savefig('Y_distribution.png')
    
    
def equal_variance_test(df):
    """ test for heteroskedasticity """
    """ to test the entire data set, pass argument as pd.concat([train,test]) """
    all_samples = df[['ProductCategory','MasterSKU','month','new_product',
                      'price_change','cluster','monthly_sum_order_qty']].values
    return bartlett(*all_samples)


def calculate_adj_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1)

def smape(Y_test, Y_pred):
    """ returns a percentage """
    return np.mean(np.abs(Y_pred-Y_test)/((np.abs(Y_test)+np.abs(Y_pred))/2)) * 100

def cross_val_smape(model, X_, Y_, fold=5, rand_seed=None):

    X_train, X_test, Y_train, Y_test = [], [], [], []
    kf = KFold(n_splits=fold, random_state=rand_seed)
    
    for train_indices, test_indices in kf.split(X_, Y_):
        X_train, X_test = X_[train_indices], X_[test_indices]
        Y_train, Y_test = Y_[train_indices], Y_[test_indices]
    
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    return smape(Y_test, Y_pred)


def aic(residual, n, p):
    SSE = np.sum(residual**2)
    score = n * np.log(SSE/n) + 2*p
    return score 

def bic(residual, n, p):
    SSE = np.sum(residual**2)
    score = n * np.log(SSE/n) + np.log(n)*p
    return score


def preprocess(X_train, X_test, Y_train, Y_test, boxcox_transform):
    
    """ scale and center X; box-cox transform Y """
    """ called in each modeling function """ 
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    train_shape, test_shape = Y_train.shape, Y_test.shape

    if boxcox_transform:
        Y_train, best_lambda = boxcox(Y_train)
        Y_test = boxcox(Y_test, lmbda=best_lambda)
        if train_shape != Y_train.shape or test_shape != Y_test.shape:
            raise ValueError("Check Box-Cox transformation lambda")

    return X_train, X_test, Y_train, Y_test


##########################################################################################
################### Ordinary Least Square Regression #####################################
##########################################################################################


def ols_ttest(X_train, Y_train, boxcox_transform=False):
    
    """ returns statsmodels summary with p-values for OLS linear regression """
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)

    X_train = sm.add_constant(X_train)
    model = sm.OLS(Y_train, X_train)
    
    result = model.fit()
    print("Warning: Pearson correlation and t-test/p-value below are unreliable if data is heteroskedastic")
    print("Call equal_variance_test() to check for heteroskedasticity")
    
    return result.summary()


def ols(X_train, X_test, Y_train, Y_test, boxcox_transform=True, plot=True, by_masterSKU=False, return_values=False):

    """ Use this function for modeling with train/test split. Use ols_predict(X_train, Y_train) to predict future values """
    
    X_train, X_test, Y_train, Y_test = preprocess(X_train, X_test, Y_train, Y_test, boxcox_transform=boxcox_transform)
    
    model = LinearRegression()
    
    # cross validate on training data
    n_train = X_train.shape[0]        # num of training observations
    p = X_train.shape[1]              # num of features
    rmse_train = np.mean(np.sqrt((-1 * cross_val_score(model,X_train,Y_train,
                                        scoring='neg_mean_squared_error',
                                        cv=5))))
    r2_list = cross_val_score(model, X_train, Y_train, scoring='r2')
    adj_r2_train = np.mean([calculate_adj_r2(r2, n_train, p) for r2 in r2_list])
    smape_train = np.mean(cross_val_smape(model, X_train, Y_train))
    
    # fit & predict test data
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # evaluate test performance
    rmse_test = np.sqrt(mean_squared_error(Y_test,Y_pred))
    smape_test = smape(Y_test, Y_pred)
    
    # plot residuals
    if plot:
        title = "Linear Regression residuals\nBox-Cox transformation: {}".format(boxcox_transform)
        plot_residuals(Y_test, Y_pred, title)
    
    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        train_summary = """
        Ordinary Least Square Regression, Box-Cox transformation: {}
        Training RMSE: {}
        Training Adjusted R2: {}
        Training SMAPE: {}%""".format(boxcox_transform, rmse_train, adj_r2_train, smape_train)
        print(train_summary)
        print("\nTest RMSE: {}, test SMAPE: {}%".format(rmse_test, smape_test))
    if return_values:
        op = smape_test, Y_pred
        
    return op

def plot_residuals(Y_test, Y_pred, title):
    """ plot residuals of fit models """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=.2)
    ax.set_ylabel('residual')
    ax.set_xlabel('y-hat')
    ax.set_title(title)
    residuals = Y_test - Y_pred
    ax.scatter(Y_pred, residuals, alpha=0.7)
    # fig.savefig('residuals.png')

def ols_by_masterSKU(train, test, return_values=False, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):
 
    """ return per-masterSKU results in 2 dicts """
    rmse, smape, Y_pred = {}, {}, {}
    X_train_all, X_test_all, Y_train_all, Y_test_all = split_train_test(train, test, feature_list=feature_list)
    
    j = 0
    
    for i, masterSKU in enumerate(ENCODED_MASTERSKUS):
        
        X_test, Y_test = split_train_test(train, test, masterSKU=i, feature_list=feature_list)
        
        if return_values:
            smape[masterSKU], Y_pred[masterSKU] = ols(X_train_all, X_test, Y_train_all, Y_test,
                                                    plot=False, by_masterSKU=True, return_values=True)
        else:
            rmse[masterSKU], smape[masterSKU] = ols(X_train_all, X_test, Y_train_all, Y_test,
                                                    plot=False, by_masterSKU=True)
  
        sys.stdout.write('\r'+"{0:.3%}".format(j/len(ENCODED_MASTERSKUS)))
        sys.stdout.flush()
        j += 1
        
    if return_values:
        op = rmse, smape
    else:
        op = smape, Y_pred
    return op



def polynomial_ols(X_train, X_test, Y_train, Y_test, d, boxcox_transform=True,plot=True,by_masterSKU=False,return_values=False):

    """ Use this function for modeling with train/test split """
    """ Use polynomial_ols_predict(X_train, Y_train) to predict future values """
    
    if d > 4:
        raise ValueError("Use degree <= 4")
    
    X_train, X_test, Y_train, Y_test = preprocess(X_train, X_test, Y_train, Y_test, boxcox_transform=boxcox_transform)
    
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    
    # cross validate on training data
    n_train = X_train.shape[0]      # num of training observations
    p = X_train.shape[1]            # num of features
    rmse_train = np.mean(np.sqrt((-1 * cross_val_score(model,X_train,Y_train,
                                        scoring='neg_mean_squared_error',
                                        cv=5))))
    r2_list = cross_val_score(model, X_train, Y_train, scoring='r2')
    adj_r2_train = np.mean([calculate_adj_r2(r2, n_train, p) for r2 in r2_list])
    smape_train = cross_val_smape(model, X_train, Y_train)
    
    # fit & predict test data
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # test performance 
    rmse_test = np.sqrt(mean_squared_error(Y_test,Y_pred))
    smape_test = smape(Y_test, Y_pred)
    
    # plot residuals
    if plot:
        title = "Polynomial Linear Regression residuals\nBox-Cox transformation: {}".format(boxcox_transform)
        plot_residuals(Y_test, Y_pred, title)
    
    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        train_summary = """
        Polynomial Linear Regression, Box-Cox transformation: {}
        Training RMSE: {}
        Training Adjusted R2: {}
        Training SMAPE: {}%""".format(boxcox_transform, rmse_train, adj_r2_train, smape_train)
        print(train_summary)
        print("\nTest RMSE: {}, test SMAPE: {}%".format(rmse_test,smape_test))
        
    if return_values:
        op = smape_test, Y_pred
    
    return op


def polynomial_ols_by_masterSKU(train, test, d, return_values=False, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):
    
    """ return per-masterSKU results in 2 dicts """
    
    rmse, smape, Y_pred = {}, {}, {}
    X_train_all, X_test_all, Y_train_all, Y_test_all = split_train_test(train, test, feature_list=feature_list)
    
    j = 0
    
    for i, masterSKU in enumerate(ENCODED_MASTERSKUS):
        
        X_test, Y_test = split_train_test(train, test, masterSKU=i, feature_list=feature_list)
        
        if return_values:
            smape[masterSKU], Y_pred[masterSKU] = polynomial_ols(X_train_all, X_test, Y_train_all, Y_test,
                                                           d=d, plot=False, by_masterSKU=True, return_values=True)
        else:
            rmse[masterSKU], smape[masterSKU] = polynomial_ols(X_train_all, X_test, Y_train_all, Y_test,
                                                           d=d, plot=False, by_masterSKU=True)
        
        sys.stdout.write('\r'+"{0:.3%}".format(j/len(ENCODED_MASTERSKUS)))
        sys.stdout.flush()
        j += 1
    
    if return_values:
        op = rmse, smape
    else:
        op = smape, Y_pred
        
    return op
    


##########################################################################################
######################## Weighted Least Square Regression ################################
##########################################################################################

def wls(X_train, X_test, Y_train, Y_test, w=None, plot=True, boxcox_transform=True, by_masterSKU=False, return_values=False):
    
    """ Use this function for modeling with train/test split. Use wls_predict(X_train, Y_train) to predict future values """
    """ default w: more recent records = greater weights """
    
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocess(X_train, X_test, Y_train, Y_test,
                                                  boxcox_transform=boxcox_transform)
    if w is None:
        w = np.asarray([1/(i+1) for i in range(X_train.shape[0])][::-1])

    # fit and predict
    model = LinearRegression()
    model.fit(X_train, Y_train, sample_weight=w)
    Y_pred = model.predict(X_test)
    
    # evaluate test performance
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred))
    smape_test = smape(Y_test, Y_pred)
    
    # plot residuals
    if plot:
        title = "Weighted Least Square Regression Residuals\nBox-Cox transformation: {}".format(boxcox_transform)
        plot_residuals(Y_test, Y_pred, title)
    
    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        print("\n Test RMSE: {}, test SMAPE: {}%".format(rmse_test,smape_test))
    if return_values:
        op = smape_test, Y_pred

    return op


def wls_by_masterSKU(train, test, w=None, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):
    
    """ return per-masterSKU results in 2 dicts """
    
    rmse, smape, Y_pred = {}, {}, {}
    X_train_all, X_test_all, Y_train_all, Y_test_all = split_train_test(train, test, feature_list=feature_list)

    j = 0
    for i, masterSKU in enumerate(ENCODED_MASTERSKUS):
        
        X_test, Y_test = split_train_test(train, test, masterSKU=i, feature_list=feature_list)
        
        if return_values:
            smape[masterSKU], Y_pred[masterSKU] = wls(X_train_all, X_test, Y_train_all, Y_test, w=w,
                                                    plot=False, by_masterSKU=True, return_values=True)
        else:
            rmse[masterSKU], smape[masterSKU] = wls(X_train_all, X_test, Y_train_all, Y_test, w=w, 
                                                    plot=False, by_masterSKU=True)
        
        sys.stdout.write('\r'+"{0:.3%}".format(j/len(ENCODED_MASTERSKUS)))
        sys.stdout.flush()
        j += 1
        
    if return_values:
        op = rmse, smape
    else:
        op = smape, Y_pred

    return op


##########################################################################################
################ KNN regression ##########################################################
##########################################################################################

def knn_find_k(X_train, Y_train, lo=2, hi=70, step=1, boxcox_transform=True, return_best_k=True):
    
    """ find the optimal K using a sample of the training data set by best cross-validated RMSE on training data """
    """ reduce run time: set step=2 or greater & use training data from split_train_test(frac=0.2) """
    
    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
    if hi > len(X_train):
        hi = len(X_train) / 2
        
    print("Job start time:", datetime.datetime.now())

    r2_list, rmse_list = [], []
    n = X_train.shape[0]
    p = X_train.shape[1]

    for k in range(lo, hi, step):
        
        model = KNeighborsRegressor(n_neighbors = k)
        
        try:
            r2 = np.mean(cross_val_score(model, X_train, Y_train, scoring='r2'))
            r2_list.append(r2)
            rmse_train = np.mean(np.sqrt((-1 * cross_val_score(model,X_train,Y_train,
                                        scoring='neg_mean_squared_error',
                                        cv=5))))
            rmse_list.append(rmse_train)
        except:
            ValueError("Increase training data set size")
        
        sys.stdout.write('\r'+"{0:.3%}".format((k-lo)/hi * 1/step))
        sys.stdout.flush()
        
    adj_r2_list = [calculate_adj_r2(r2, n, p) for r2 in r2_list]
    print("\nJob finish time:", datetime.datetime.now())
    
    if return_best_k:
        op = np.argmin(rmse_list)*2 
    else:
        op = adj_r2_list, rmse_list
    return op



def plot_knn_results(lo, hi, score_list, step=1):
    """ step param = knn_find_k(step) param """
    fig, ax = plt.subplots()
    X, Y = [k for k in range(lo, hi, step)], score_list
    ax.plot(X, Y)
    plt.grid(color='grey',linestyle='-', linewidth=1, alpha=.2)
    ax.set_title("KNN regressor ({} <= k < {})".format(lo, hi))
    ax.set_xlabel("number of neighbors")
    ax.set_ylabel("scores")
    # fig.savefig('knn_results.png')

def knn(X_train, X_test, Y_train, Y_test, k, plot=True, boxcox_transform=True, by_masterSKU=False, return_values=False):
    
    """ Use this function for modeling with train/test split. Use knn_predict(X_train, Y_train) to predict future values """
    """ the model has already cross-validated on training data when selecting K """
    
    X_train, X_test, Y_train, Y_test = preprocess(X_train, X_test, Y_train, Y_test, boxcox_transform=boxcox_transform)
    
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # evaluate test performance
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred))
    smape_test = smape(Y_test, Y_pred)

    if plot:
        title = "KNN Regressor (K = {}) residuals\nBox-Cox transformation: {}".format(k,boxcox_transform)
        plot_residuals(Y_test, Y_pred, title)

    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        print("\n Test RMSE: {}, test SMAPE: {}%".format(rmse_test, smape_test))
        
    if return_values:
        op = smape_test, Y_pred
    
    return op




def knn_by_masterSKU(train, test, k, return_values=False, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):

    """ return per-masterSKU results in 2 dicts """
    
    rmse, smape, Y_pred = {}, {}, {}
    X_train_all, X_test_all, Y_train_all, Y_test_all = split_train_test(train, test, feature_list=feature_list)
    
    j = 0
    
    for i, masterSKU in enumerate(ENCODED_MASTERSKUS):
        
        X_test, Y_test = split_train_test(train, test, masterSKU=i, feature_list=feature_list)
        
        if return_values:
            smape[masterSKU], Y_pred[masterSKU] = knn(X_train_all, X_test, Y_train_all, Y_test, k=k, 
                                                    plot=False, by_masterSKU=True, return_values=True)
        else:
            rmse[masterSKU], smape[masterSKU] = knn(X_train_all, X_test, Y_train_all, Y_test, k=k, 
                                                    plot=False, by_masterSKU=True)
        
        sys.stdout.write('\r'+"{0:.3%}".format(j/len(ENCODED_MASTERSKUS)))
        sys.stdout.flush()
        j += 1
        
    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        print("\n Test RMSE: {}, test SMAPE: {}%".format(rmse_test,smape_test))
        
    return op


##########################################################################################
######################## Random forest regressor #########################################
##########################################################################################


def random_forest(X_train, X_test, Y_train, Y_test, tree=150, plot=True, boxcox_transform=True, by_masterSKU=False, return_values=False):
    
    """ Use this function for modeling with train/test split """ 
    """ Use random_forest_predict(X_train, Y_train) to predict future values """

    X_train, X_test, Y_train, Y_test = preprocess(X_train, X_test, Y_train, Y_test, boxcox_transform=boxcox_transform)
    model = RandomForestRegressor(n_estimators=tree, oob_score=True,
                                  bootstrap=True,
                                  max_depth=None,
                                  n_jobs=-1)

    # cross validate on training data
    if not by_masterSKU:
        train_rmse = np.mean(-1 * cross_val_score(model,X_train,Y_train,scoring='neg_mean_squared_error',cv=5))
        train_smape = np.mean(cross_val_smape(model, X_train, Y_train))
  
        
        train_summary = """
            Random Forest, Box-Cox transformation: {}
            Training RMSE: {}
            Training OOB: {}
            Training SMAPE :{}%
        """.format(boxcox_transform, train_rmse, model.oob_score_, train_smape)
        print(train_summary)
    else:
        model = model.fit(X_train, Y_train)
    
    model = model.fit(X_train, Y_train)
    
    # evaluate test performance
    Y_pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred))
    smape_test = smape(Y_test, Y_pred)
    
    if plot:
        title = "Random Forest ({} trees) residuals\nBox-Cox transformation: {}".format(tree,boxcox_transform)
        plot_residuals(Y_test, Y_pred, title)
    
    if by_masterSKU:
        op = rmse_test, smape_test
    else:
        op = model
        print("\nTest RMSE: {}, test SMAPE: {}%".format(rmse_test, smape_test))
    if return_values:
        op = smape_test, Y_pred
        
    return op



def plot_oob_scores(tree_list, score_list):
    fig4, ax4 = plt.subplots()
    ax4.plot(tree_list, score_list)
    plt.grid(color='grey',linestyle='-', linewidth=1, alpha=.2)
    ax4.set_title("Random forest estimated R2 per # of trees")
    ax4.set_xlabel("Number of trees")
    ax4.set_ylabel("Estimated R2")
    # fig.savefig("rf_oob.pgn")

    
    
def find_optimal_tree(X_train, Y_train, tree_list=[100,150,200,250,300,350,400,450], boxcox_transform=True, plot=False):
    
    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
    oob_list = []
    
    for n in tree_list:
        model = RandomForestRegressor(n_estimators=n, oob_score=True,
                                  bootstrap=True,
                                  max_depth=None,
                                  n_jobs=-1)
        model.fit(X_train, Y_train)
        oob_list.append(model.oob_score_)
    
    if plot:
        plot_oob_scores(tree_list, oob_list)
    best_tree = (np.argmax(oob_list)+2)*50     # change this if tree_list is not default
    
    return best_tree


    
def random_forest_by_masterSKU(train, test, tree=150, return_values=False, feature_list=['ProductCategory','MasterSKU','month','new_product','price_change','cluster']):

    """ return per-masterSKU results in 2 dicts """
    
    rmse, smape, Y_pred = {}, {}, {}
    X_train_all, X_test_all, Y_train_all, Y_test_all = split_train_test(train, test, feature_list=feature_list)
    
    j = 0
    for i, masterSKU in enumerate(ENCODED_MASTERSKUS):
        X_test, Y_test = split_train_test(train, test, masterSKU=i, feature_list=feature_list)
        if return_values:
            rmse[masterSKU], smape[masterSKU] = random_forest(X_train_all, X_test, Y_train_all, Y_test,
                                                              tree=tree, plot=False, by_masterSKU=True, return_values=True)
        else:
            rmse[masterSKU], smape[masterSKU] = random_forest(X_train_all, X_test, Y_train_all, Y_test,
                                                              tree=tree, plot=False, by_masterSKU=True)
        sys.stdout.write('\r'+"{0:.3%}".format(j/len(ENCODED_MASTERSKUS)))
        sys.stdout.flush()
        j += 1
    
    if return_values:
        op = smape, Y_pred
    else:
        op = rmse, smape
    return op


##########################################################################################
##################### Prediction (100% training data) ####################################
##########################################################################################

def predict_(model, month, new_product=False, price_change=None):
    
    """ takes a fit model and returns monthly order sum for each masterSKU in a dict """
    
    # format inputs
    month = float(month)
    new_product_flag = float(1) if new_product else float(0)
    price_change = float(price_change) if price_change is not None else float(0)
    prediction = {}
    
    for masterSKU_id in range(18):
        
        for cluster in range(5):
            
            masterSKU = ENCODED_MASTERSKUS[masterSKU_id]
            category = float(find_product_category(masterSKU_id))
            masterSKU_id = float(masterSKU_id)
            cluster = float(cluster)
            exog = np.asarray([category,masterSKU_id,month,new_product,price_change,cluster]).reshape(1,6)
            
            if masterSKU not in prediction:
                prediction[masterSKU] = 0
                
            prediction[masterSKU] += model.predict(exog)
    
    return prediction



def ols_predict(X_train, Y_train, month, new_product=False, price_change=None, boxcox_transform=True):
    """ returns predicted value in nparray """

    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
        
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    result = predict_(model, month, new_product=new_product, price_change=price_change)
    return result


def wls_predict(X_train, Y_train, month, new_product=False, price_change=None, w=None, boxcox_transform=True):
    
    """ returns predicted value in nparray """

    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
    if w is None:
        w = np.asarray([1/(i+1) for i in range(X_train.shape[0])][::-1])
    model = LinearRegression()
    model.fit(X_train, Y_train, sample_weight=w)
    
    result = predict(model, month, new_product=new_product, price_change=price_change)
    return result


def polynomial_ols_predict(X_train, Y_train, d, month, new_product=False, price_change=None, boxcox_transform=True):
    
    """ returns predicted value in nparray """
    
    if d > 4:
        raise ValueError("Use degree <= 4")
    
    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
        
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, Y_train)
    
    result = predict(model, month, new_product=new_product, price_change=price_change)
    return result



def knn_predict(X_train, Y_train, k, month, new_product=False, price_change=None, boxcox_transform=True):
    
    """ returns predicted value in nparray """
    
    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
        
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, Y_train)
    
    result = predict(model, month, new_product=new_product, price_change=price_change)
    return result
    

def random_forest_predict(X_train, Y_train, month, tree=150, new_product=False, price_change=None, boxcox_transform=True):
    
    """ returns predicted value in nparray """
    
    X_train = StandardScaler().fit_transform(X_train)
    if boxcox_transform:
        Y_train, _ = boxcox(Y_train)
        
    model = RandomForestRegressor(n_estimators=tree, oob_score=True,
                                  bootstrap=True, max_depth=None,
                                  n_jobs=-1)
    model = model.fit(X_train, Y_train)
    
    result = predict(model, month, new_product=new_product, price_change=price_change)
    return result


