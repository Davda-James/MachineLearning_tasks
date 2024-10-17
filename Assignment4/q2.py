import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 


def find_correlation_with_target(feature,target):
    pcoeff=pearsonr(feature,target)[0]
    return pcoeff
def solve_pearson_question(xtrain,ytrain):
    curr=0
    for i in  xtrain.columns:
        pearsoncoeff=find_correlation_with_target(xtrain[i],ytrain)
        if pearsoncoeff>curr:
            most_related_feature=i
            curr=pearsoncoeff
    return (curr,most_related_feature)

def fit_line(xtrain,ytrain):
    xtrain_arr=np.array(xtrain.copy())
    if xtrain_arr.ndim == 1:
        xtrain_arr = xtrain_arr.reshape(-1, 1)
    ones_column = np.ones((xtrain_arr.shape[0], 1))  # Create a column of ones
    xtrain_arr = np.hstack((ones_column, xtrain_arr))  # Horizontal stack
    weights = np.dot(np.dot(np.linalg.inv(np.dot(xtrain_arr.T,xtrain_arr)),xtrain_arr.T),ytrain)
    return weights
def predict_data(xtest_feature,weights):
    y_predicted=np.dot(np.reshape(xtest_feature,(xtest_feature.shape[0],1)),weights[1:])+weights[0]
    return y_predicted

def calculate_rmse(y1,y2):  
    y1cp=np.array(y1)
    y2cp=np.array(y2)
    return np.sqrt(np.sum(((y1cp-y2cp)**2)/y1.shape[0]))    

def plot_it(x,y,weights):
    plt.plot(x,x*weights[1]+weights[0],color ='red')
    plt.show()
if __name__=='__main__':
    dftrain=pd.read_csv('abalone_train.csv')
    dftest=pd.read_csv('abalone_test.csv')
    xtrain=dftrain.drop(columns=['Rings'])
    ytrain=dftrain['Rings']
    xtest=dftest.drop(columns=['Rings'])
    ytest=dftest['Rings']
    most_realted_feature_pearsoncoeff_target,most_realted_feature =solve_pearson_question(xtrain,ytrain)
    # i
    weigths_with_single_feature=fit_line(xtrain[most_realted_feature],ytrain)
    y_predicted_test=predict_data(xtest[most_realted_feature],weigths_with_single_feature)
    y_predicted_train=predict_data(xtrain[most_realted_feature],weigths_with_single_feature)
    # plot_it(xtrain[most_realted_feature],ytrain,weigths_with_single_feature)
    # ii
    rmse_error_train=calculate_rmse(ytrain,y_predicted_train)
    # iii
    rmse_error_test=calculate_rmse(ytest,y_predicted_test)
    # print(rmse_error_train)
    # print(rmse_error_test)
    # iv
    plt.scatter(ytest,y_predicted_test,color='orange')
    plt.show()