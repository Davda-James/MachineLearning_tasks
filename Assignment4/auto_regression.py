import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

if __name__=='__main__':
    df=pd.read_csv("asianpaint.csv")
    dftrain=df[:int((0.65*df.shape[0]))+1]
    dftest=df[int((0.65*df.shape[0]))+1:]
    dftrain.to_csv('asianpaint_train.csv',index=False)
    dftest.to_csv('asianpaint_test.csv',index=False)
    dfcp=df.copy()
    dfcp.iloc[:, 0] = pd.to_datetime(dfcp.iloc[:, 0],dayfirst=True)
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.yticks(np.linspace(df.iloc[:, 1].min(), df.iloc[:, 1].max(), 30))
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Train and Test Data')
    plt.plot(np.arange(dftrain.shape[0]),dftrain.iloc[:,1],color="blue",label='train data')
    plt.plot(np.arange(dftrain.shape[0], dftrain.shape[0] + dftest.shape[0]),dftest.iloc[:,1],color="red",label='test data')
    plt.tight_layout()
    plt.legend()
    plt.show()