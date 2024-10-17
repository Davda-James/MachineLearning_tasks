import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def find_weights(x,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)

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


