import numpy as np 
import pandas as pd 
import sys
import os  
from sklearn.preprocessing import LabelEncoder
from scipy.stats import multivariate_normal
sys.path.append('')
path=os.path.join(os.path.dirname(__file__))
pathTofile = os.path.join(path,"..","Assignment2","Assignment2_p1" )
sys.path.append(pathTofile)
from p1 import pca_algo1
def apply_same_pca_test_data(x_train,x_test,k,eigen_vectors):
    x_test=x_test-x_train.mean(axis=0)  
    x_test_arr = np.array(x_test)
    x_test_proj=np.matmul(x_test_arr,eigen_vectors)
    return (x_test_arr,x_test_proj)
    
def gaussian_probability(x,mean_subtracted_x,sigma):
    return (np.exp(-0.5*((mean_subtracted_x/sigma)**2)))/(sigma)
def calculate_confusion_matrix(y_original,y_predicted,mapping):
    cm=pd.DataFrame(0,columns=[mapping[0],mapping[1],mapping[2]],index=[mapping[0],mapping[1],mapping[2]])
    for i in range(len(y_test_encoded)):
        cm.at[mapping[y_predicted[i]],mapping[y_test_encoded[i]]]+=1
    return cm
    
def model1(x_test_proj,means_class):
    n=len(x_test_proj)    
    std=means_class.loc[:,('feature','std')]
    y_predicted=np.empty(n)
    for i in range(n):
        mean_sub_from_x=x_test_proj[i]-means_class.loc[:,('feature','mean')]
        prob=gaussian_probability(x_test_proj[i],mean_sub_from_x,std)
        max_prob= prob.idxmax()
        y_predicted[i]  =max_prob
    return y_predicted
def test_bayes_classifier_multivariate(x_test,means_x_train_df,cov_x_train_df_class):
    x_test_arr=np.array(x_test)
    means_x_train_df_arr=np.array(means_x_train_df)
    cov_x_train_df_class_arr=np.array(cov_x_train_df_class)
    n_classes=means_x_train_df.shape[0]
    n_samples=x_test.shape[0]
    likelihood=np.zeros((n_samples,n_classes))
    for i in range(n_classes):
        mean=means_x_train_df_arr[i]
        cov=cov_x_train_df_class_arr[i]
        mvn=multivariate_normal(mean=mean,cov=cov)
        likelihood[:,i]=mvn.pdf(x_test_arr)
    y_predicted_mvn=np.argmax(likelihood,axis=1)
    return y_predicted_mvn

def model2_bayes_multivariate(x_train):
    x_train_df =x_train.copy()
    x_train_df['Species']=y_train_encoded
    x_train_df_grouped=x_train_df.groupby('Species')
    means_x_train_df=x_train_df_grouped.mean()
    cov_x_train_df_class=x_train_df_grouped.apply(lambda g:np.cov(g,rowvar=False),include_groups=False)
    return (means_x_train_df,cov_x_train_df_class)

def calculate_accuracy(cm,y_predicted):
    accuracy_score=sum(np.diag(cm))/len(y_predicted)
    return accuracy_score   

if __name__=='__main__':
    train_data=pd.read_csv("./iris_train.csv",index_col=False)
    test_data=pd.read_csv("./iris_test.csv",index_col=False)
    x_train=train_data.drop(columns=[train_data.columns[0],train_data.columns[len(train_data.columns)-1]],axis=1)
    y_train=train_data[train_data.columns[len(train_data.columns)-1]]
    x_test=test_data.drop(columns=[test_data.columns[0],test_data.columns[len(test_data.columns)-1]],axis=1)
    y_test=test_data[test_data.columns[len(test_data.columns)-1]]
    # question 1 i->is implemented here
    x_train_mean_sub,x_train_proj,eigen_values,eigen_vectors=pca_algo1(x_train,1)
    x_test_arr,x_test_proj= apply_same_pca_test_data(x_train,x_test,1,eigen_vectors)     
    # question 1 ii->part is implemented here 
    label_encoder = LabelEncoder()
    y_train_encoded=label_encoder.fit_transform(y_train)
    y_test_encoded=label_encoder.fit_transform(y_test)
    mapping=label_encoder.classes_
    df_train=pd.DataFrame({"feature":x_train_proj.flatten(),"species" :y_train_encoded})
    means_class =df_train.groupby('species').agg(['mean','std'])
    # question 1 iii->part 
    y_predicted=model1(x_test_proj,means_class)
    y_predicted=y_predicted.astype(int)
    #  confusion matrix
    cm_model1=calculate_confusion_matrix(y_test_encoded,y_predicted,mapping)
    #  calculation of accuracy
    score_model1=calculate_accuracy(cm_model1,y_predicted)
    # print(accuracy_score)

    #  question 2  is implemented here 
    #  question 2 part->i
    means_x_train_df,cov_x_train_df_class =model2_bayes_multivariate(x_train)
    # question2 part->>ii
    y_predicted_mvn=test_bayes_classifier_multivariate(x_test,means_x_train_df,cov_x_train_df_class)
    cm_model2=calculate_confusion_matrix(y_test_encoded,y_predicted_mvn,mapping)
    # print(cm_model2)
    score_model2=calculate_accuracy(cm_model2,y_predicted_mvn)
    
    #  question3 part -> a 
    print(score_model2-score_model1)


    