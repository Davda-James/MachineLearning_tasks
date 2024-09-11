import pandas as pd 
from math import sqrt
def calculate_stats(temp_col):
    rows= len(temp_col)
    # mean 
    mean=0
    #maximum
    maxi=0
    #minimum
    mini=500
    #standard deviaton
    std=0
    for i in range(rows):
        el=temp_col[i]
        mean+=el
        maxi=max(maxi,el)
        mini=min(mini,el)
        std+=el*el
    mean = mean/rows
    #finding standard deviation
    std = sqrt(std/rows - mean*mean)
    sorted_temp_col = sorted(temp_col)
    #findinf median
    median=0
    if rows%2==0:
        median = (sorted_temp_col[(rows-1)//2]+ sorted_temp_col[(rows-1)//2+1])/2
    else :
        median = sorted_temp_col[rows//2]
    mean =round(mean,2)
    median=round(median,2)
    std = round(std,2)
    return (mean,maxi,mini,median,std)

if __name__ == '__main__':
    df= pd.read_csv("../landslide_data_original.csv")
    mean,maxi,mini,median,std = calculate_stats(df['temperature'])
    print("The mean of temperature column is : ",mean)
    print("The maximum of temperature column is : ",maxi)
    print("The minimum of temperature column is : ",mini)
    print("The median of temperature column is : ",median)
    print("The std of temperature column is : ",std)
    # print(df["temperature"].describe()) can crossverify using the inbuilt stats data 

