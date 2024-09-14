import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split as tts 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from collections import Counter 
def df_boxplot(attribute_df,cols):
    fig,axs=plt.subplots(2,2)
    for i in range(len(cols)):
        axs[i//2][i%2].boxplot(attribute_df[cols[i]])
        axs[i//2][i%2].set_title(cols[i])
    plt.show()  


def find_outliers_and_replace_with_median(attribute_df):
    def find_percentile(index1, index2, weight):
        return sorted_col[index1] + weight * (sorted_col[index2] - sorted_col[index1])

    total_len = attribute_df.shape[0]
    sorted_col = sorted(attribute_df['SepalWidthCm'])
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
    attribute_df.loc[(attribute_df['SepalWidthCm']>interquartile_range_upperbound) | (attribute_df['SepalWidthCm']<interquartile_range_lowerbound),'SepalWidthCm'] = sorted_col[total_len//2]


def pca_algo1(attribute_df):
    attribute_df=attribute_df-attribute_df.mean()
    X=np.array(attribute_df)
    X_transpose = X.T
    cov_matrix = np.matmul(X_transpose,X)/(X.shape[0] - 1) 
    eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
    # print(eigen_values) 
    args_sorted = np.argsort(eigen_values)[::-1]
    choosen_eigen_values = eigen_values[args_sorted][:2]
    choosen_eigen_vector=eigen_vectors[:,args_sorted][:,:2]
    X_proj = np.matmul(X,choosen_eigen_vector)
    return (X,X_proj,choosen_eigen_values,choosen_eigen_vector)

def plot_xtrans_with_eigen_vectors(X_trans,choosen_eigen_vectors):
    #  plotting starts here 
    plt.scatter(X_trans[:,0],X_trans[:,1],alpha=0.8,label='Data Points')
    origin = np.mean(X_trans,axis=0)
    scaling_factor=2
    for i in range(choosen_eigen_vectors.shape[1]):
        scaled_eigenvector = choosen_eigen_vectors[:2,i] * scaling_factor
        plt.quiver(origin[0],origin[1], scaled_eigenvector[0]  , scaled_eigenvector[1],
            angles='xy', scale_units='xy', scale=1, color=['r', 'b'][i],
            label=f'Eigenvector {i+1}')
    plt.axhline(0, color='grey', linewidth=0.01)
    plt.axvline(0, color='grey', linewidth=0.01)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot with Eigen Directions')
    plt.legend()
    plt.grid(True)
    plt.show()

def test(X,X_trans):
    pca=PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # print(X_pca)
    # print(X_pca==X_trans)
    # Plot the original data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5)
# Get eigenvectors and eigenvalues
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_ratio_
    print(eigenvectors)
    # Plot the eigenvectors (directions of maximum variance)

def reversing_pca(X_trans,eigen_vectors):
    X_frompca =   np.matmul(X_trans,eigen_vectors.T)
    return X_frompca

def calculate_rmse(X,X_frompca):
    temp = np.sqrt(np.mean((X-X_frompca)**2,axis=0))
    return temp

def knn(x_train,y_train,x_test):
    k=5 
    y_predict=np.zeros((len(x_test),),dtype=int)
    for i in range(len(x_test)):
        dist = np.zeros((len(x_train),2),dtype=(float))
        for j in range(len(x_train)):
            dist[j][0]=np.sqrt((x_test[i][0]-x_train[j][0])**2+(x_test[i][1]-x_train[j][1])**2)
            dist[j][1]=y_train[j]
        dist=sorted(dist,key =lambda x:x[0])[:k]
        second_values = [x[1] for x in dist]
        count= Counter(second_values)
        y_predict[i]=count.most_common(1)[0][0]
    return y_predict

def calculate_confusion_matrix(y_predict,y):
    confMatrix = confusion_matrix(y,y_predict)
    return confMatrix
def display_confusion(confMatrix):
    disp = ConfusionMatrixDisplay(confusion_matrix=confMatrix,display_labels =[0,1,2])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.show()
    

if __name__=="__main__":
    df = pd.read_csv("../Iris.csv")
    attribute_df = df.drop(columns=['Species'])
    target_df = df['Species']
    y=np.array(target_df)
    le = LabelEncoder()
    y=le.fit_transform(y)
    cols=attribute_df.columns
    #  df_boxplot(attribute_df,cols)
    #  outliers removed from the data
    find_outliers_and_replace_with_median(attribute_df)
    #  df_boxplot(attribute_df,cols)
    #  converting to nd array X
    X,X_trans,eigen_values,choosen_eigen_vectors =pca_algo1(attribute_df)
    # print(choosen_eigen_vectors)
    #  plotting a scatter plot of transformed data 
    
    # plot_xtrans_with_eigen_vectors(X_trans,choosen_eigen_vectors)
    
    # print(choosen_eigen_vectors[:, 0])
    # test(X,X_trans)
    # print(X_trans)
    
    #  get the data back from pca (original mean centered)
    X_frompca = reversing_pca(X_trans,choosen_eigen_vectors)
    rmse_arr = calculate_rmse(X,X_frompca)
    #  uncomment this to get the arry of the rmse values of respective columns of features
    print(rmse_arr) 

    #  here starts knn algorithm 
    x_train,x_test,y_train,y_test = tts(X_trans,y,random_state=104,test_size=0.2,shuffle=True)
    y_predict =knn(x_train,y_train,x_test)
    confMatrix = calculate_confusion_matrix(y_predict,y_test)
    ''' printing the confusion matrix '''
    #  print(confMatrix)
    ''' displaying confusion matrix '''
    display_confusion(confMatrix)






