import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

def plot_boxplot_columns(df,df_cols):
    fig,axs = plt.subplots(2,4)  
    axs = axs.flatten()
    for i in range(len(df_cols)):
        axs[i].boxplot(df[df_cols[i]])
        axs[i].set_title(f"{df_cols[i]}")
    for j in range(7, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
# def find_percentile(sorted_col,total_len):
#     if total_len%4==0 :
#         per25 = sorted_col[total_len//4] 
#         per75 = sorted_col[3*(total_len//4)]
#     else:
#         per25 = sorted_col[total_len//4-1] + (total_len/4-total_len//4) * (sorted_col[total_len//4] - sorted_col[total_len//4-1])
#         per75 = sorted_col[(3*total_len)//4-1] + ((3*total_len)/4-(3*total_len)//4) * (sorted_col[(3*total_len)//4] - sorted_col[(3*total_len)//4-1])
#     return (per25,per75)
def find_outliers_and_plot(df_copy,df_cols):
    def find_percentile(index1, index2, weight):
        return sorted_col[index1] + weight * (sorted_col[index2] - sorted_col[index1])
    
    total_len = df_copy.shape[0]
    for i in df_cols:
        sorted_col = sorted(df_copy[i])
        p25_pos = 0.25 * (total_len - 1)
        p75_pos = 0.75 * (total_len - 1)
        p25_index1 = int(p25_pos)
        p25_index2 = min(p25_index1 + 1, total_len - 1) 
        p75_index1 = int(p75_pos)
        p75_index2 = min(p75_index1 + 1, total_len - 1)
        p25_weight = p25_pos - p25_index1
        p75_weight = p75_pos - p75_index1
        per25 = find_percentile(p25_index1, p25_index2, p25_weight)
        per75 = find_percentile(p75_index1, p75_index2, p75_weight)
        interquartile_range_upperbound= per75+1.5*(per75-per25)
        interquartile_range_lowerbound = per25-1.5*(per75-per25)
        df_copy.loc[(df_copy[i]>interquartile_range_upperbound) | (df_copy[i]<interquartile_range_lowerbound),i] = sorted_col[total_len//2]
    df_copy.to_csv("../landslide_data_removed_outliers.csv",index=False)
    plot_boxplot_columns(df_copy,df_cols)   

if __name__ == "__main__":
    df = pd.read_csv("../landslide_data_interpolated.csv",index_col=False)
    df_cols = df.columns 
    plot_boxplot_columns(df,df_cols)
    # yes from the graph we can clearly see there is a need of feature scaling and there are outliers right now in this 
    df_copy = df.copy()
    ''' this is 3rd problem question 2'''
    find_outliers_and_plot(df_copy,df_cols)    

