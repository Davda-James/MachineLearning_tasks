import pandas as pd 
import matplotlib.pyplot as plt 
import numpy  as np 

            
if __name__ =="__main__":
    df = pd.read_csv("../landslide_data_original.csv")
    humidity_data_t12 = df[df['stationid']=="t12"]['humidity']
    bin_size= 5
    # calculating min and max of humidity data in station id t12
    min_humidity = int(humidity_data_t12.min()//bin_size*bin_size)
    max_humidity = int(humidity_data_t12.max()//bin_size * bin_size)+bin_size
    #  calcualation of bins
    bins = np.arange(min_humidity,max_humidity+bin_size,bin_size)
    # bins array
    counts=np.zeros(len(bins)-1)
    #  distributing each value inside the range which they should lie 
    for value in humidity_data_t12:
        i = (value-min_humidity) // bin_size 
        if(i<len(counts)):
            counts[int(i)]+=1
    #  plot starts here 
    plt.title("Histogram plot of humidity values from t12 station")
    plt.xlabel("Range of humidity values")
    plt.ylabel("Frequency distribution of humidity")
    plt.bar(bins[:-1], counts, width=bin_size, edgecolor='black', align='edge')
    plt.tight_layout()
    plt.show() 