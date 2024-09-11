import pandas as pd 
import os 
import sys 
from math import sqrt
import matplotlib.pyplot as plt 
from numpy import mean  
df = pd.read_csv("../landslide_data_miss.csv")
# intially there are 19 rows which have null values in stationid

def find_next_non_null_value(idx,col_data):
    rows=df.shape[0]
    for i in range(idx,rows):
        if(pd.isna(col_data[i])==False):
            return i
        
def do_interpolation(col,col_name,rows):
    prev=0
    for i in range(1,rows):
        if(pd.isna(col[i])==True):
            idx = find_next_non_null_value(i+1,col)
            col[i] = col[idx] + (i-prev)*((col[idx]-col[prev])/(idx-prev))
        else:
            prev=i
    df[col_name]=col

def calculate_stats_original_new(original_df):
    df_stats = pd.DataFrame(index=['mean','maximum','minimum','median','std'],columns=df.columns)
    sys.path.append(os.path.abspath(os.path.join('..',"Assignment1_p1_B23123")))
    from Assignment1_p1_q1_B23123 import calculate_stats
    for i in cols:
        mean_latest,maxi_latest,mini_latest,median_latest,std_latest=calculate_stats(df[i])
        mean_original,maxi_original,mini_original,median_original,std_original = calculate_stats(original_df[i])
        df_stats.at['mean',i] = (mean_latest,mean_original)
        df_stats.at['maximum',i] = (maxi_latest,maxi_original)
        df_stats.at['minimum',i] = (mini_latest,mini_original)
        df_stats.at['median',i] = (median_latest,median_original)
        df_stats.at['std',i] = (std_latest,std_original)
    return df_stats

def compare_rmse(original_df,df,cols,rows):
    rmse = []  
    indexes = df.index.to_list()
    for i in cols:
        val=0
        for j in indexes:
            val+=(original_df.at[j,i]-df_without_reset_index.at[j,i])**2
        rmse.append(sqrt(val/rows))
    return rmse
def plot_rmse_comparisons(original_df,df,cols,rows):
    y = compare_rmse(original_df,df,cols,rows)
    plt.title("RMSE")
    plt.xlabel("Features")
    plt.ylabel("RMSE between original and interpolated")
    plt.scatter(cols,y,color="cyan")
    plt.show()

if __name__ == "__main__":
    ''' this is problem 2 question 1 '''
    # removing  the rows with null values in stationid column
    # we can see that rows have been removed as the shape has been decreased after and before 
    print(df.shape)
    df.drop(columns=['stationid','dates'],inplace=True)
    cols= df.columns
    df.dropna(thresh=(2*len(cols))//3,inplace=True)
    original_index= df.index
    df.reset_index(drop=True,inplace=True)
    print(df.shape) #final shape 

    ''' this is problem 2 question 2 part a'''

    rows = df.shape[0]
    for i in cols:
        do_interpolation(df[i],i,rows) 
    df_without_reset_index = df.copy()
    df_without_reset_index.index = original_index
    df.to_csv("../landslide_data_interpolated.csv",index=False) 
    original_df=pd.read_csv("../landslide_data_original.csv")
    original_df.drop(columns=['stationid','dates'],inplace=True) 
    print(calculate_stats_original_new(original_df))
    # print(original_df)
    print(df.isnull().sum()) # this is to check whether any null values are there or not we got 0 null values 

    ''' this is the problem 2 question 2 part b'''
    # this is the plot of root mean square error of each cols of interpolated data with new one
    plot_rmse_comparisons(original_df,df_without_reset_index,cols,rows)
    # print(compare_rmse(original_df,df_without_reset_index,cols,rows))
