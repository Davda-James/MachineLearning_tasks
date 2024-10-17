from sklearn.model_selection import train_test_split as tts
import pandas as pd 

if __name__=='__main__':
    df=pd.read_csv("./abalone.csv")
    x_train,x_test,y_train,y_test=tts(df.drop(columns=[df.columns[len(df.columns)-1]]),df[df.columns[len(df.columns)-1]],test_size=0.3,random_state=42)
    x_train['Rings']=y_train.values 
    x_test['Rings']=y_test.values   
    x_train.to_csv('abalone_train.csv')
    x_test.to_csv('abalone_test.csv')
    