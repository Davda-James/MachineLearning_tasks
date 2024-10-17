import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def find_weights(x,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
def calculate_rmse(y1,y2):  
    target_range=max(y1)-min(y1)
    rmse=np.sqrt(np.sum(((y1-y2)**2)/y1.shape[0]))   
    return (1.0 - (rmse / target_range)) * 100
def calculate_mape(y1,y2):
    return 100*np.mean(abs(ytest_actual-ytest_predicted)/ytest_actual)

if __name__=='__main__':
    dftrain=pd.read_csv('asianpaint_train.csv')
    dftest=pd.read_csv('asianpaint_test.csv')
    #  creating the lag
    dftraincp=dftrain.copy()
    dftestcp=dftest.copy()
    dftraincp['Price_lag1']=dftraincp['Open'].shift(1)
    dftestcp['Price_lag1']=dftestcp['Open'].shift(1)
    dftraincp.dropna(inplace=True)
    dftestcp.dropna(inplace=True)
    y=dftraincp['Open'].values.astype(float)
    x=dftraincp['Price_lag1'].values.astype(float)
    xtest=dftestcp['Price_lag1'].values.astype(float)
    ytest=dftestcp['Open'].values.astype(float)
    x=np.column_stack([np.ones(x.shape[0]),x])
    weights=find_weights(x,y)
    ytest_actual=dftestcp['Open']    
    ytest_predicted=weights[0]+weights[1]*xtest
    plt.figure(figsize=(10, 6))
    plt.plot(dftestcp.index, ytest_actual, label='Actual Prices', color='blue')
    plt.plot(dftestcp.index, ytest_predicted, label='Predicted Prices', color='red', linestyle='--')
    plt.title('Actual vs Predicted Stock Prices (One-step ahead)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    rmse=calculate_rmse(ytest_actual,ytest_predicted)
    print(rmse)
    mape=calculate_mape(ytest_actual,ytest_predicted)
    print(mape) 

    

