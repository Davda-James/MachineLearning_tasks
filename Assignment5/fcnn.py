import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import save_model, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def load_image(path):
    images=[]
    labels=[]
    for i,class_name in enumerate(['0','1','2']):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path=os.path.join(class_path,img_name)
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img) / 255.0
                images.append(img)
                labels.append(i)
    images=np.array(images)
    labels=np.array(labels)
    return images,labels
    

if __name__=='__main__':
    img_size=(32,32)
    train_dir='./cifar-3class-data/cifar-3class-data/train'
    test_dir ='./cifar-3class-data/cifar-3class-data/test'
    xtrain,ytrain=load_image(train_dir)
    xtest,ytest=load_image(test_dir)
    print(xtrain.shape)
    print(ytrain.shape)
    plt.figure(figsize=(6,10))
    fig,axes = plt.subplots(nrows=2,ncols=2)
    for i in range(4):
        axes[i//2][i%2].imshow(xtrain[i])
        axes[i//2][i%2].set_title(f'Label: {ytrain[i]}')
    plt.tight_layout()
    plt.show()

    xtrain,xval,ytrain,yval = tts(xtrain,ytrain,random_state=29,test_size =0.1)
    print(xtrain.shape)
    print(xval.shape)
    xtrain_flatten=xtrain.reshape(xtrain.shape[0],-1)
    xval_flatten=xval.reshape(xval.shape[0],-1)
    xtest_flatten=xtest.reshape(xtest.shape[0],-1)
    print(xtrain_flatten.shape)
    print(ytrain.shape)
    print(xval_flatten.shape)
    print(yval.shape)
    print(xtest_flatten.shape)
    # defining the model
    model = Sequential()
    model.add(Dense(256,activation='relu',input_shape=(3072,)))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    fcnn_history=model.fit(xtrain_flatten,ytrain,epochs=500,batch_size=200,validation_data=(xval_flatten,yval))
    # save the model fcnn
    save_dir = './saved_model'
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'my_fcnn_model.keras'))
    with open('./saved_model/history.pkl', 'wb') as f:
        pickle.dump(fcnn_history.history, f)
    ypred=model.predict(xtest_flatten)
    ypred=np.argmax(ypred,axis=1)   
    score_fcnn=accuracy_score(ytest,ypred)
    print(score_fcnn)



   