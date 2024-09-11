import pandas as pd
from numpy import mean,std


def manual_minmax_scaler(df,df_cols,upper,lower):
    for i in df_cols:
        df[i] = lower + (upper-lower)*((df[i]-df[i].min())/(df[i].max()-df[i].min()))

def manual_std_scaler(df,df_cols,df_ans):
    for i in df_cols: 
        original_mean = df[i].mean()
        original_std = df[i].std()
        df[i]=(df[i]-df[i].mean())/df[i].std()
        df_ans.at['mean',i]=(round(df[i].mean(),3),original_mean)
        df_ans.at['std',i]=(round(df[i].std(),3),original_std)
    
if __name__ == "__main__":
    df = pd.read_csv("../landslide_data_removed_outliers.csv")
    df_copy=df.copy()
    df_cols = df.columns  
    df_ans = pd.DataFrame(index=['mean','std'],columns=df_cols)
    # initial_outlier_removed_data_mean = mean(df) 
    upper=12
    lower=5
    ''' this is problem 4 question 1'''
    manual_minmax_scaler(df,df_cols,upper,lower)
    print(df.min())
    print(df.max())  
    # print(df)
    # we will get 5 and 12 
    
    ''' this is problem 4 question 2'''
    manual_std_scaler(df_copy,df_cols,df_ans)
    print(df_ans)



