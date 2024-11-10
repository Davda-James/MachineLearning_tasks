import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import seaborn as sns
import os
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

def load_image(path):
    images=[]
    labels=[]
    for i,class_name in enumerate(['0','1','2']):
        class_path = os.path.join(path, class_name)
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
#  cnn model starts here 
    cnn_model=Sequential()
    cnn_model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
    cnn_model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
    cnn_model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    cnn_model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512,activation='relu'))
    cnn_model.add(Dense(100,activation='relu'))
    cnn_model.add(Dense(3,activation='softmax'))
    cnn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    cnn_history=cnn_model.fit(xtrain,ytrain,epochs=50,batch_size=200,validation_data=(xval,yval))
    save_dir = './saved_model'
    cnn_model.save(os.path.join(save_dir, 'my_cnn_model.keras'))
    with open('./saved_model/cnn_history.pkl','wb') as f:
        pickle.dump(cnn_history.history,f)
    ypred=cnn_model.predict(xtest)
    ypred=np.argmax(ypred,axis=1)
    cnn_score=accuracy_score(ytest,ypred)
    print(cnn_score)

    accuracies_epochs = cnn_history['accuracy']
    val_accuracies=cnn_history['val_accuracy']
    epochs = range(1,len(accuracies_epochs)+1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies_epochs, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red', marker='x')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    conf_matrix=confusion_matrix(ytest,ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
    
