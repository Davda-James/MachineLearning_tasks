import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def polynomial_features(X,degree):
    n_samples,n_features=X.shape
    polyx = np.ones((n_samples,1))
    for d in range(1,degree):
        polyx= np.hstack((polyx,X**d))
    return polyx

def fit_curve(X,y,degree):
    poly_x=polynomial_features(X,degree)
    weights=np.dot(np.dot(np.linalg.inv(np.dot(poly_x.T,poly_x)),poly_x.T),y)
    return weights
def predict_y(xtest,poly_weights,degree):
    poly_xtest=polynomial_features(xtest,degree)
    return np.dot(poly_xtest,poly_weights)

def calculate_rmse(y1,y2):  
    return np.sqrt(np.sum(((y1-y2)**2)/y1.shape[0]))    


if __name__=='__main__':
    dftrain=pd.read_csv('abalone_train.csv')
    dftest=pd.read_csv('abalone_test.csv')
    xtrain=dftrain['Shell weight']
    ytrain=dftrain['Rings']
    xtest=dftest["Shell weight"]
    ytest=dftest['Rings']
    xtrain_arr=np.array(xtrain).reshape((xtrain.shape[0],1))
    ytrain_arr=np.array(ytrain)
    xtest_arr=np.array(xtest).reshape((xtest.shape[0],1))
    ytest_arr=np.array(ytest)

    # try out for degree 2 3 4 5 
    train_rmse=np.zeros(4)
    test_rmse=np.zeros(4)
    most_least_rmse=2000
    final_weights=None
    final_train_predicted=None 
    for i in range(2,6):
        poly_weights=fit_curve(xtrain_arr,ytrain_arr,i)
        y_train_predicted=predict_y(xtrain_arr,poly_weights,i)
        train_rmse[i-2]=calculate_rmse(ytrain,y_train_predicted)
        y_predicted=predict_y(xtest_arr,poly_weights,i)
        test_rmse[i-2]=calculate_rmse(ytest_arr,y_predicted)
        if test_rmse[i-2]<most_least_rmse:
            most_least_rmse=test_rmse[i-2]
            final_train_predicted=y_train_predicted
            final_weights=poly_weights
    min_rmse_degree=np.argmin(test_rmse)+2
    print(min_rmse_degree)
    # print(train_rmse)
    # print(test_rmse)
    # #  ques-i  plotting the bar graph for train
    plt.xlabel("Degree")
    plt.ylabel("RMSE")
    plt.title("degree vs rmse")
    plt.bar(np.arange(2,6),train_rmse)
    plt.show()


    #  ques-ii  plotting the bar graph for test
    plt.xlabel("Degree")
    plt.ylabel("RMSE")
    plt.title("degree vs rmse")
    plt.bar(np.arange(2,6),test_rmse)
    plt.show()
    
    # ques-iii 
    sorted_indices = np.argsort(xtrain_arr[:, 0])
    plt.plot(xtrain_arr[sorted_indices], final_train_predicted[sorted_indices], color="red", label="Best Fit Curve")
    plt.scatter(xtrain_arr, ytrain_arr, color="blue", label="Training Data")
    plt.xlabel("Shell Weight")
    plt.ylabel("Rings")
    plt.title(f"Best Fit Curve (Degree {min_rmse_degree})")
    plt.legend()
    plt.show()
