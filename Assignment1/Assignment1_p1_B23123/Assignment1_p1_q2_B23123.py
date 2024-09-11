import pandas as pd 

def pearson_math(x,y):
    x_mean = x.mean()
    y_mean = y.mean()
    std_x = x.std()
    std_y = y.std()
    return (((x-x_mean)*(y-y_mean)).sum())/(len(x)*std_x*std_y)


def calculate_pearson_corelation(data):
    final_df_cols = data.columns
    final_df =  pd.DataFrame(-1.0,index=final_df_cols,columns=final_df_cols)
    for i in final_df_cols:
        for j in final_df_cols:
            final_df.at[i,j] = pearson_math(df[i],df[j])
    return final_df
            
if __name__ =="__main__":
    df = pd.read_csv("../landslide_data_original.csv")
    data = df.drop(columns=['dates','stationid'])
    pearson_relation = calculate_pearson_corelation(data)
    print(pearson_relation)
